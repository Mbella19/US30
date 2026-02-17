#!/usr/bin/env python3
"""
Feature importance analysis for the PPO Agent's observation vector.

Runs 8 data-driven analyses on all 37 MARKET_FEATURE_COLS to determine
which features are predictive, redundant, unstable, or noise.

Analyses:
1. Statistical Profiling (sparsity, clipping, distribution shape)
2. Correlation / Redundancy Matrix
3. Predictive Power (Mutual Information + Spearman)
4. Temporal Stability (PSI + KS test)
5. Cross-Timeframe Redundancy
6. Random Forest Feature Importance
7. Feature Uniqueness (conditional value for correlated pairs)
8. Marginal Group Value

Usage:
    python scripts/analyze_features_for_agent.py
    python scripts/analyze_features_for_agent.py --sample-size 30000 --skip-cross-tf
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.live.bridge_constants import MARKET_FEATURE_COLS

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_GROUPS: Dict[str, List[str]] = {
    "regime_volatility_raw": ["atr", "chop", "adx"],
    "trend": ["sma_distance"],
    "market_structure": ["dist_to_support", "dist_to_resistance", "sr_strength_r", "sr_strength_s"],
    "sessions": ["session_asian", "session_london", "session_ny"],
    "structure_breaks": ["structure_fade", "bars_since_bos", "bars_since_choch", "bos_magnitude", "choch_magnitude"],
    "price_dynamics": ["returns", "volatility"],
    "volatility_context": ["atr_context"],
    "regime_robust_percentiles": [
        "atr_percentile", "chop_percentile",
        "sma_distance_percentile", "volatility_percentile",
    ],
    "ood_v36_deprecated": ["volatility_regime", "distribution_shift_score"],
    "ood_v37": [
        "volatility_vs_training", "returns_skew_shift",
        "atr_vs_training", "range_vs_training", "ood_score",
    ],
    "mean_reversion": [
        "bb_percent_b", "bb_bandwidth", "price_zscore", "williams_r",
        "rsi", "cci", "rsi_divergence",
    ],
}

BINARY_FEATURES = {
    "session_asian", "session_london", "session_ny",
}
TERNARY_FEATURES = set()
DISCRETE_FEATURES = BINARY_FEATURES | TERNARY_FEATURES | {"volatility_regime"}

HORIZONS = [12, 24, 48, 96]  # 1h, 2h, 4h, 8h at 5m bars
TRAIN_END_DATE = "2025-07-01"
VAL_END_DATE = "2025-11-01"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _fmt(x: float, d: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{x:.{d}f}"


def _feature_group(feat: str) -> str:
    for grp, cols in FEATURE_GROUPS.items():
        if feat in cols:
            return grp
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Data Loading (with re-engineering from OHLCV if needed)
# ─────────────────────────────────────────────────────────────────────────────

def _reengineer_features(processed_dir: Path) -> None:
    """Re-engineer features from existing OHLCV parquets using updated feature code."""
    from src.data.features import engineer_all_features
    from src.data.resampler import align_timeframes
    from src.data.ood_features import TrainingBaseline
    from src.data.normalizer import FeatureNormalizer
    from config.settings import Config

    config = Config()

    logger.info("Re-engineering features from existing OHLCV data...")

    # Check if raw features already have new columns (just need normalization)
    df_check = pd.read_parquet(processed_dir / "features_5m.parquet", columns=None)
    raw_already_done = "structure_fade" in df_check.columns
    del df_check

    if raw_already_done:
        logger.info("  Raw features already updated — just re-normalizing...")
        df_5m = pd.read_parquet(processed_dir / "features_5m.parquet")
        df_15m = pd.read_parquet(processed_dir / "features_15m.parquet")
        df_45m = pd.read_parquet(processed_dir / "features_45m.parquet")

        # Skip to normalization
        raw_cols = ['open', 'high', 'low', 'close', 'atr', 'chop', 'adx',
                    'session_asian', 'session_london', 'session_ny', 'atr_context']
        train_end = pd.Timestamp(TRAIN_END_DATE)
        train_mask_full = df_5m.index < train_end
        normalize_cols = [c for c in df_5m.columns if c not in raw_cols and c != 'volume']

        for label, df in [("5m", df_5m), ("15m", df_15m), ("45m", df_45m)]:
            cols_to_norm = [c for c in normalize_cols if c in df.columns]
            normalizer = FeatureNormalizer(feature_cols=cols_to_norm)
            normalizer.fit(df.loc[train_mask_full])
            df_norm = df.copy()
            df_norm[cols_to_norm] = normalizer.transform(df)[cols_to_norm]
            df_norm.to_parquet(processed_dir / f"features_{label}_normalized.parquet")

        logger.info("  Normalization complete!")
        return

    # Load existing parquets (they have OHLCV + old features)
    df_5m = pd.read_parquet(processed_dir / "features_5m.parquet",
                            columns=["open", "high", "low", "close"])
    df_15m = pd.read_parquet(processed_dir / "features_15m.parquet",
                             columns=["open", "high", "low", "close"])
    df_45m = pd.read_parquet(processed_dir / "features_45m.parquet",
                             columns=["open", "high", "low", "close"])

    # Add volume if missing (needed by some features)
    for df in [df_5m, df_15m, df_45m]:
        if "volume" not in df.columns:
            df["volume"] = 1000.0

    logger.info(f"  5m: {len(df_5m):,}  15m: {len(df_15m):,}  45m: {len(df_45m):,}")

    feature_config = {
        'fractal_window': config.features.fractal_window,
        'sr_lookback': config.features.sr_lookback,
        'sma_period': config.features.sma_period,
        'chop_period': config.features.chop_period,
        'adx_period': config.features.adx_period,
        'atr_period': config.features.atr_period,
    }

    # Pass 1: Basic features to compute training baseline
    logger.info("  Pass 1: Computing basic features for training baseline...")
    df_5m_p1 = engineer_all_features(df_5m.copy(), feature_config)
    df_15m_p1 = engineer_all_features(df_15m.copy(), feature_config)
    df_45m_p1 = engineer_all_features(df_45m.copy(), feature_config)

    df_5m_t, df_15m_t, df_45m_t = align_timeframes(df_5m_p1, df_15m_p1, df_45m_p1)

    valid = ~(df_5m_t.isna().any(axis=1) | df_15m_t.isna().any(axis=1) | df_45m_t.isna().any(axis=1))
    df_5m_valid = df_5m_t[valid]

    train_end = pd.Timestamp(TRAIN_END_DATE)
    train_mask = df_5m_valid.index < train_end
    train_df = df_5m_valid[train_mask]
    logger.info(f"  Training baseline from {len(train_df):,} samples")

    training_baseline = TrainingBaseline.from_training_data(train_df)
    del df_5m_p1, df_15m_p1, df_45m_p1, df_5m_t, df_15m_t, df_45m_t, df_5m_valid
    gc.collect()

    # Pass 2: Full features with training baseline
    logger.info("  Pass 2: Re-computing features WITH training baseline...")
    df_5m = engineer_all_features(df_5m, feature_config, training_baseline=training_baseline)
    df_15m = engineer_all_features(df_15m, feature_config, training_baseline=training_baseline)
    df_45m = engineer_all_features(df_45m, feature_config, training_baseline=training_baseline)

    df_5m, df_15m, df_45m = align_timeframes(df_5m, df_15m, df_45m)

    # Drop NaN rows across all timeframes
    valid = ~(df_5m.isna().any(axis=1) | df_15m.isna().any(axis=1) | df_45m.isna().any(axis=1))
    df_5m = df_5m[valid]
    df_15m = df_15m[valid]
    df_45m = df_45m[valid]

    logger.info(f"  Final aligned rows: {len(df_5m):,}")
    logger.info(f"  Features: {[c for c in df_5m.columns if c in MARKET_FEATURE_COLS]}")

    # Save raw features
    df_5m.to_parquet(processed_dir / "features_5m.parquet")
    df_15m.to_parquet(processed_dir / "features_15m.parquet")
    df_45m.to_parquet(processed_dir / "features_45m.parquet")

    # Normalize
    logger.info("  Normalizing features...")
    raw_cols = ['open', 'high', 'low', 'close', 'atr', 'chop', 'adx',
                'session_asian', 'session_london', 'session_ny', 'atr_context']

    train_mask_full = df_5m.index < train_end
    normalize_cols = [c for c in df_5m.columns if c not in raw_cols and c != 'volume']

    for label, df in [("5m", df_5m), ("15m", df_15m), ("45m", df_45m)]:
        cols_to_norm = [c for c in normalize_cols if c in df.columns]
        normalizer = FeatureNormalizer(feature_cols=cols_to_norm)
        normalizer.fit(df.loc[train_mask_full])
        df_norm = df.copy()
        df_norm[cols_to_norm] = normalizer.transform(df)[cols_to_norm]
        df_norm.to_parquet(processed_dir / f"features_{label}_normalized.parquet")

    logger.info("  Feature re-engineering complete!")


def load_data(processed_dir: Path, force_reengineer: bool = False) -> dict:
    """Load processed parquets with only needed columns."""
    logger.info("Loading data...")

    # Check if features need re-engineering (old features present)
    df_check = pd.read_parquet(processed_dir / "features_5m.parquet", columns=None)
    needs_reengineer_raw = "structure_fade" not in df_check.columns or force_reengineer
    del df_check

    # Also check if normalized parquets have new features
    df_check_norm = pd.read_parquet(processed_dir / "features_5m_normalized.parquet", columns=None)
    needs_reengineer_norm = "structure_fade" not in df_check_norm.columns
    del df_check_norm

    needs_reengineer = needs_reengineer_raw or needs_reengineer_norm

    if needs_reengineer:
        logger.info("Detected OLD features in parquets — re-engineering with updated code...")
        _reengineer_features(processed_dir)

    # Columns we need
    need_cols = sorted(set(MARKET_FEATURE_COLS + ["close"]))

    df_5m_raw = pd.read_parquet(
        processed_dir / "features_5m.parquet",
        columns=[c for c in need_cols if c != "close"]
        + ["close", "open", "high", "low"],
    )
    df_5m_norm = pd.read_parquet(
        processed_dir / "features_5m_normalized.parquet",
        columns=need_cols,
    )
    df_15m_norm = pd.read_parquet(
        processed_dir / "features_15m_normalized.parquet",
        columns=[c for c in need_cols if c in pd.read_parquet(
            processed_dir / "features_15m_normalized.parquet", columns=[]).columns],
    )
    df_45m_norm = pd.read_parquet(
        processed_dir / "features_45m_normalized.parquet",
        columns=[c for c in need_cols if c in pd.read_parquet(
            processed_dir / "features_45m_normalized.parquet", columns=[]).columns],
    )

    # Date-based masks
    if isinstance(df_5m_norm.index, pd.DatetimeIndex):
        train_mask = df_5m_norm.index < TRAIN_END_DATE
        val_mask = (df_5m_norm.index >= TRAIN_END_DATE) & (df_5m_norm.index < VAL_END_DATE)
    else:
        n = len(df_5m_norm)
        split = int(0.85 * n)
        train_mask = pd.Series(False, index=df_5m_norm.index)
        train_mask.iloc[:split] = True
        val_mask = ~train_mask

    logger.info(
        f"  5m rows: {len(df_5m_norm):,}  "
        f"train: {train_mask.sum():,}  val: {val_mask.sum():,}"
    )

    return {
        "df_5m_raw": df_5m_raw,
        "df_5m_norm": df_5m_norm,
        "df_15m_norm": df_15m_norm,
        "df_45m_norm": df_45m_norm,
        "train_mask": train_mask,
        "val_mask": val_mask,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Target Construction
# ─────────────────────────────────────────────────────────────────────────────

def compute_future_returns(close: pd.Series, horizons: List[int]) -> pd.DataFrame:
    """Compute future returns and direction at multiple horizons."""
    targets = pd.DataFrame(index=close.index)
    for h in horizons:
        fret = (close.shift(-h) / close - 1.0).astype(np.float32)
        targets[f"fret_{h}"] = fret
        targets[f"fdir_{h}"] = np.sign(fret).astype(np.float32)
    return targets


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Statistical Profiling
# ─────────────────────────────────────────────────────────────────────────────

def statistical_profile(
    df_raw: pd.DataFrame, df_norm: pd.DataFrame, cols: List[str]
) -> pd.DataFrame:
    """Compute distribution stats for each feature (raw + normalized)."""
    logger.info("Running statistical profiling...")
    rows = []
    for col in cols:
        row: Dict[str, object] = {"feature": col, "group": _feature_group(col)}

        for suffix, df in [("raw", df_raw), ("norm", df_norm)]:
            if col not in df.columns:
                continue
            v = df[col].dropna().to_numpy(dtype=np.float64)
            n = len(v)
            if n == 0:
                continue
            row[f"mean_{suffix}"] = float(np.mean(v))
            row[f"std_{suffix}"] = float(np.std(v))
            row[f"min_{suffix}"] = float(np.min(v))
            row[f"max_{suffix}"] = float(np.max(v))
            row[f"skew_{suffix}"] = float(pd.Series(v).skew())
            row[f"kurtosis_{suffix}"] = float(pd.Series(v).kurtosis())
            row[f"zero_frac_{suffix}"] = float((v == 0).mean())
            row[f"near_zero_frac_{suffix}"] = float((np.abs(v) < 0.001).mean())
            row[f"unique_{suffix}"] = int(pd.Series(v).nunique())
            if suffix == "norm":
                row["clip_frac_norm"] = float((np.abs(v) >= 4.999).mean())
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Correlation / Redundancy
# ─────────────────────────────────────────────────────────────────────────────

def correlation_analysis(
    df_norm: pd.DataFrame, cols: List[str], train_mask: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson correlation matrix + flagged pairs."""
    logger.info("Running correlation analysis...")
    avail = [c for c in cols if c in df_norm.columns]
    train_df = df_norm.loc[train_mask, avail].dropna()
    corr = train_df.corr(method="pearson")

    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = []
    for i in range(len(avail)):
        for j in range(i + 1, len(avail)):
            r = float(upper.iloc[i, j])
            ar = abs(r)
            if ar >= 0.70:
                tier = (
                    "near_duplicate_0.95" if ar >= 0.95
                    else "very_high_0.85" if ar >= 0.85
                    else "high_0.70"
                )
                pairs.append({
                    "feature_a": avail[i],
                    "feature_b": avail[j],
                    "pearson_r": r,
                    "abs_r": ar,
                    "tier": tier,
                })

    pairs_df = pd.DataFrame(pairs).sort_values("abs_r", ascending=False) if pairs else pd.DataFrame()
    return pairs_df, corr


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Predictive Power (MI + Spearman)
# ─────────────────────────────────────────────────────────────────────────────

def predictive_power(
    df_norm: pd.DataFrame,
    targets: pd.DataFrame,
    cols: List[str],
    train_mask: pd.Series,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Mutual information and Spearman correlation with future returns."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from scipy.stats import spearmanr

    logger.info("Running predictive power analysis (MI + Spearman)...")
    avail = [c for c in cols if c in df_norm.columns]
    discrete_mask = np.array([c in DISCRETE_FEATURES for c in avail])

    merged = pd.concat([df_norm[avail], targets], axis=1)
    merged = merged.loc[train_mask].dropna()

    if len(merged) > sample_size:
        merged = merged.sample(n=sample_size, random_state=seed)
    logger.info(f"  Using {len(merged):,} samples for MI/Spearman")

    X = merged[avail].to_numpy(dtype=np.float64)
    results: Dict[str, Dict[str, float]] = {c: {"feature": c} for c in avail}

    for h in HORIZONS:
        y_reg = merged[f"fret_{h}"].to_numpy(dtype=np.float64)
        y_dir = merged[f"fdir_{h}"].to_numpy(dtype=np.float64)
        dir_mask = y_dir != 0
        X_dir = X[dir_mask]
        y_dir_binary = (y_dir[dir_mask] > 0).astype(int)

        mi_reg = mutual_info_regression(
            X, y_reg, discrete_features=discrete_mask, n_neighbors=3, random_state=seed
        )
        mi_cls = mutual_info_classif(
            X_dir, y_dir_binary, discrete_features=discrete_mask[: X_dir.shape[1]]
            if X_dir.shape[1] == len(discrete_mask) else discrete_mask,
            n_neighbors=3, random_state=seed,
        )

        for i, col in enumerate(avail):
            results[col][f"mi_reg_{h}"] = float(mi_reg[i])
            results[col][f"mi_cls_{h}"] = float(mi_cls[i])

            rho, pval = spearmanr(X[:, i], y_reg)
            results[col][f"spearman_{h}"] = float(rho) if not np.isnan(rho) else 0.0
            results[col][f"spearman_p_{h}"] = float(pval) if not np.isnan(pval) else 1.0

        logger.info(f"  Horizon {h} bars done")
        gc.collect()

    df = pd.DataFrame(list(results.values()))
    mi_reg_cols = [f"mi_reg_{h}" for h in HORIZONS]
    mi_cls_cols = [f"mi_cls_{h}" for h in HORIZONS]
    df["mi_reg_avg"] = df[mi_reg_cols].mean(axis=1)
    df["mi_cls_avg"] = df[mi_cls_cols].mean(axis=1)
    return df.sort_values("mi_reg_avg", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Temporal Stability
# ─────────────────────────────────────────────────────────────────────────────

def _psi(train_vals: np.ndarray, val_vals: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index."""
    uniq = np.unique(train_vals)
    if len(uniq) <= 3:
        bins = np.append(uniq, uniq[-1] + 1)
        n_bins = len(uniq)
    else:
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(train_vals, percentiles)
        bins = np.unique(bins)
        if len(bins) < 3:
            return 0.0

    train_hist, _ = np.histogram(train_vals, bins=bins)
    val_hist, _ = np.histogram(val_vals, bins=bins)

    eps = 1e-6
    train_pct = train_hist / train_hist.sum() + eps
    val_pct = val_hist / val_hist.sum() + eps

    psi_val = float(np.sum((val_pct - train_pct) * np.log(val_pct / train_pct)))
    return max(psi_val, 0.0)


def temporal_stability(
    df_norm: pd.DataFrame,
    cols: List[str],
    train_mask: pd.Series,
    val_mask: pd.Series,
) -> pd.DataFrame:
    """Measure distribution shift between train and validation."""
    from scipy.stats import ks_2samp

    logger.info("Running temporal stability analysis...")
    rows = []
    for col in cols:
        if col not in df_norm.columns:
            continue
        train_v = df_norm.loc[train_mask, col].dropna().to_numpy(dtype=np.float64)
        val_v = df_norm.loc[val_mask, col].dropna().to_numpy(dtype=np.float64)
        if len(train_v) < 100 or len(val_v) < 100:
            continue

        ks_stat, ks_p = ks_2samp(train_v, val_v)
        psi_val = _psi(train_v, val_v)
        mean_shift = (np.mean(val_v) - np.mean(train_v)) / (np.std(train_v) + 1e-10)

        rows.append({
            "feature": col,
            "mean_train": float(np.mean(train_v)),
            "std_train": float(np.std(train_v)),
            "mean_val": float(np.mean(val_v)),
            "std_val": float(np.std(val_v)),
            "psi": psi_val,
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "mean_shift_stds": float(mean_shift),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("psi", ascending=False) if not df.empty else df


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Cross-Timeframe Redundancy
# ─────────────────────────────────────────────────────────────────────────────

def cross_timeframe_redundancy(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_45m: pd.DataFrame,
    cols: List[str],
    train_mask: pd.Series,
    sample_size: int,
) -> pd.DataFrame:
    """Correlate same feature across 5m/15m/45m timeframes."""
    logger.info("Running cross-timeframe redundancy analysis...")
    rows = []
    for col in cols:
        if col not in df_5m.columns or col not in df_15m.columns or col not in df_45m.columns:
            continue

        idx = np.where(train_mask.to_numpy())[0]
        if len(idx) > sample_size:
            rng = np.random.default_rng(42)
            idx = rng.choice(idx, size=sample_size, replace=False)

        v5 = df_5m[col].iloc[idx].to_numpy(dtype=np.float64)
        v15 = df_15m[col].iloc[idx].to_numpy(dtype=np.float64)
        v45 = df_45m[col].iloc[idx].to_numpy(dtype=np.float64)

        valid = ~(np.isnan(v5) | np.isnan(v15) | np.isnan(v45))
        v5, v15, v45 = v5[valid], v15[valid], v45[valid]
        if len(v5) < 100:
            continue

        r_5_15 = float(np.corrcoef(v5, v15)[0, 1])
        r_5_45 = float(np.corrcoef(v5, v45)[0, 1])
        r_15_45 = float(np.corrcoef(v15, v45)[0, 1])

        redundant = abs(r_5_15) > 0.90 or abs(r_5_45) > 0.90

        rows.append({
            "feature": col,
            "corr_5m_15m": r_5_15,
            "corr_5m_45m": r_5_45,
            "corr_15m_45m": r_15_45,
            "redundant_flag": redundant,
        })

    return pd.DataFrame(rows).sort_values("corr_5m_15m", ascending=False) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Random Forest Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def rf_importance(
    df_norm: pd.DataFrame,
    targets: pd.DataFrame,
    cols: List[str],
    train_mask: pd.Series,
    sample_size: int,
    n_trees: int,
    seed: int,
) -> pd.DataFrame:
    """Random Forest Gini + permutation importance."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance as sklearn_perm_importance

    logger.info("Running Random Forest importance...")
    avail = [c for c in cols if c in df_norm.columns]

    merged = pd.concat([df_norm[avail], targets[["fdir_24"]]], axis=1)
    merged = merged.loc[train_mask].dropna()
    merged = merged[merged["fdir_24"] != 0]
    merged["fdir_24"] = (merged["fdir_24"] > 0).astype(int)

    if len(merged) > sample_size:
        merged = merged.sample(n=sample_size, random_state=seed)

    split = int(0.8 * len(merged))
    train_sub = merged.iloc[:split]
    test_sub = merged.iloc[split:]

    X_train = train_sub[avail].to_numpy(dtype=np.float32)
    y_train = train_sub["fdir_24"].to_numpy()
    X_test = test_sub[avail].to_numpy(dtype=np.float32)
    y_test = test_sub["fdir_24"].to_numpy()

    logger.info(f"  RF train: {len(train_sub):,}  test: {len(test_sub):,}")

    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=8,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_train, y_train)

    gini = rf.feature_importances_

    acc_train = float(rf.score(X_train, y_train))
    acc_test = float(rf.score(X_test, y_test))
    logger.info(f"  RF accuracy: train={acc_train:.4f}  test={acc_test:.4f}")

    perm = sklearn_perm_importance(rf, X_test, y_test, n_repeats=3, random_state=seed, n_jobs=-1)

    rows = []
    for i, col in enumerate(avail):
        rows.append({
            "feature": col,
            "gini_importance": float(gini[i]),
            "perm_importance_mean": float(perm.importances_mean[i]),
            "perm_importance_std": float(perm.importances_std[i]),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("perm_importance_mean", ascending=False)
    df["rank_gini"] = df["gini_importance"].rank(ascending=False).astype(int)
    df["rank_perm"] = df["perm_importance_mean"].rank(ascending=False).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Feature Uniqueness
# ─────────────────────────────────────────────────────────────────────────────

def feature_uniqueness(
    df_norm: pd.DataFrame,
    targets: pd.DataFrame,
    cols: List[str],
    corr_pairs: pd.DataFrame,
    train_mask: pd.Series,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """For highly correlated pairs, measure unique contribution of each member."""
    from sklearn.ensemble import RandomForestClassifier

    if corr_pairs.empty:
        return pd.DataFrame()

    logger.info("Running feature uniqueness analysis...")
    high_pairs = corr_pairs[corr_pairs["abs_r"] >= 0.85]
    if high_pairs.empty:
        logger.info("  No pairs with |r| >= 0.85, skipping")
        return pd.DataFrame()

    avail = [c for c in cols if c in df_norm.columns]
    merged = pd.concat([df_norm[avail], targets[["fdir_24"]]], axis=1)
    merged = merged.loc[train_mask].dropna()
    merged = merged[merged["fdir_24"] != 0]
    merged["fdir_24"] = (merged["fdir_24"] > 0).astype(int)

    if len(merged) > sample_size:
        merged = merged.sample(n=sample_size, random_state=seed)

    split = int(0.8 * len(merged))
    test_sub = merged.iloc[split:]
    y_test = test_sub["fdir_24"].to_numpy()

    def _rf_score(feature_list: List[str]) -> float:
        fl = [f for f in feature_list if f in merged.columns]
        if not fl:
            return 0.5
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=50,
            n_jobs=-1, random_state=seed,
        )
        rf.fit(
            merged.iloc[:split][fl].to_numpy(dtype=np.float32),
            merged.iloc[:split]["fdir_24"].to_numpy(),
        )
        return float(rf.score(test_sub[fl].to_numpy(dtype=np.float32), y_test))

    rows = []
    seen = set()
    for _, pair in high_pairs.iterrows():
        fa, fb = str(pair["feature_a"]), str(pair["feature_b"])
        key = tuple(sorted([fa, fb]))
        if key in seen:
            continue
        seen.add(key)

        score_a_only = _rf_score([fa])
        score_b_only = _rf_score([fb])
        score_both = _rf_score([fa, fb])

        rows.append({
            "feature_a": fa,
            "feature_b": fb,
            "corr": float(pair["pearson_r"]),
            "score_a_only": score_a_only,
            "score_b_only": score_b_only,
            "score_both": score_both,
            "delta_adding_b": score_both - score_a_only,
            "delta_adding_a": score_both - score_b_only,
            "recommend_keep": fa if score_a_only >= score_b_only else fb,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Section 10: Marginal Group Value
# ─────────────────────────────────────────────────────────────────────────────

def marginal_group_value(
    df_norm: pd.DataFrame,
    targets: pd.DataFrame,
    cols: List[str],
    train_mask: pd.Series,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Measure each feature group's marginal contribution via RF accuracy."""
    from sklearn.ensemble import RandomForestClassifier

    logger.info("Running marginal group value analysis...")
    avail = [c for c in cols if c in df_norm.columns]

    merged = pd.concat([df_norm[avail], targets[["fdir_24"]]], axis=1)
    merged = merged.loc[train_mask].dropna()
    merged = merged[merged["fdir_24"] != 0]
    merged["fdir_24"] = (merged["fdir_24"] > 0).astype(int)

    if len(merged) > sample_size:
        merged = merged.sample(n=sample_size, random_state=seed)

    split = int(0.8 * len(merged))
    train_sub = merged.iloc[:split]
    test_sub = merged.iloc[split:]
    y_train = train_sub["fdir_24"].to_numpy()
    y_test = test_sub["fdir_24"].to_numpy()

    def _rf_acc(feature_list: List[str]) -> float:
        fl = [f for f in feature_list if f in merged.columns]
        if not fl:
            return 0.5
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_leaf=50,
            n_jobs=-1, random_state=seed,
        )
        rf.fit(train_sub[fl].to_numpy(dtype=np.float32), y_train)
        return float(rf.score(test_sub[fl].to_numpy(dtype=np.float32), y_test))

    full_acc = _rf_acc(avail)
    logger.info(f"  Full feature set accuracy: {full_acc:.4f}")

    rows = []
    for grp, grp_cols in FEATURE_GROUPS.items():
        grp_avail = [c for c in grp_cols if c in avail]
        if not grp_avail:
            continue

        without = [c for c in avail if c not in grp_avail]
        acc_without = _rf_acc(without)
        acc_only = _rf_acc(grp_avail)
        marginal = full_acc - acc_without

        rows.append({
            "group": grp,
            "n_features": len(grp_avail),
            "features": ", ".join(grp_avail),
            "acc_full": full_acc,
            "acc_without_group": acc_without,
            "acc_group_only": acc_only,
            "marginal_accuracy": marginal,
        })
        logger.info(f"  {grp}: marginal={marginal:+.4f}  only={acc_only:.4f}")

    return pd.DataFrame(rows).sort_values("marginal_accuracy", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Section 11: Composite Ranking
# ─────────────────────────────────────────────────────────────────────────────

def composite_ranking(
    pred_power: pd.DataFrame,
    rf_imp: pd.DataFrame,
    stability: pd.DataFrame,
    stats: pd.DataFrame,
    cols: List[str],
) -> pd.DataFrame:
    """Combine analyses into a single ranked feature list."""
    logger.info("Computing composite ranking...")
    rows = []

    for col in cols:
        row: Dict[str, object] = {"feature": col, "group": _feature_group(col)}

        mi_row = pred_power[pred_power["feature"] == col]
        mi_avg = float(mi_row["mi_reg_avg"].iloc[0]) if not mi_row.empty else 0.0
        row["mi_reg_avg"] = mi_avg

        rf_row = rf_imp[rf_imp["feature"] == col] if rf_imp is not None else pd.DataFrame()
        perm_imp = float(rf_row["perm_importance_mean"].iloc[0]) if not rf_row.empty else 0.0
        row["perm_importance"] = perm_imp

        stab_row = stability[stability["feature"] == col]
        psi_val = float(stab_row["psi"].iloc[0]) if not stab_row.empty else 0.0
        row["psi"] = psi_val

        stat_row = stats[stats["feature"] == col]
        zero_frac = float(stat_row["zero_frac_raw"].iloc[0]) if not stat_row.empty and "zero_frac_raw" in stat_row.columns else 0.0
        row["zero_frac"] = zero_frac

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    n = len(df)
    df["rank_mi"] = df["mi_reg_avg"].rank(ascending=False)
    df["rank_rf"] = df["perm_importance"].rank(ascending=False)
    df["rank_stability"] = df["psi"].rank(ascending=True)

    df["composite_rank"] = (
        0.35 * df["rank_mi"]
        + 0.30 * df["rank_rf"]
        + 0.20 * df["rank_stability"]
        + 0.15 * df["zero_frac"].rank(ascending=True)
    )
    df = df.sort_values("composite_rank")

    def _recommend(r):
        rank_pct = r["composite_rank"] / n
        if r["zero_frac"] > 0.95:
            return "REMOVE (>95% zeros)"
        if r["psi"] > 0.25:
            return "UNSTABLE (PSI>0.25)"
        if rank_pct <= 0.40:
            return "KEEP"
        if rank_pct <= 0.65:
            return "KEEP (marginal)"
        return "CONSIDER_REMOVING"

    df["recommendation"] = df.apply(_recommend, axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section 12: Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    stats: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    pred_power: pd.DataFrame,
    stability: pd.DataFrame,
    cross_tf: Optional[pd.DataFrame],
    rf_imp: Optional[pd.DataFrame],
    uniqueness: Optional[pd.DataFrame],
    marginal: Optional[pd.DataFrame],
    ranking: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write CSV artifacts and markdown report."""
    logger.info("Generating report...")
    _ensure_dir(output_dir)

    stats.to_csv(output_dir / "statistical_profile.csv", index=False)
    if not corr_pairs.empty:
        corr_pairs.to_csv(output_dir / "correlation_pairs.csv", index=False)
    pred_power.to_csv(output_dir / "predictive_power.csv", index=False)
    stability.to_csv(output_dir / "temporal_stability.csv", index=False)
    if cross_tf is not None and not cross_tf.empty:
        cross_tf.to_csv(output_dir / "cross_timeframe_redundancy.csv", index=False)
    if rf_imp is not None:
        rf_imp.to_csv(output_dir / "rf_importance.csv", index=False)
    if uniqueness is not None and not uniqueness.empty:
        uniqueness.to_csv(output_dir / "feature_uniqueness.csv", index=False)
    if marginal is not None:
        marginal.to_csv(output_dir / "marginal_group_value.csv", index=False)
    ranking.to_csv(output_dir / "final_ranking.csv", index=False)

    lines: List[str] = []
    lines.append("# Feature Importance Analysis Report — US30 (Agent-Focused)")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")

    n_keep = len(ranking[ranking["recommendation"].str.startswith("KEEP")])
    n_remove = len(ranking[~ranking["recommendation"].str.startswith("KEEP")])
    lines.append(f"- **{n_keep}** features recommended to KEEP")
    lines.append(f"- **{n_remove}** features flagged for removal or review")
    lines.append("")

    lines.append("## Top 15 Features (Composite Ranking)")
    lines.append("")
    lines.append("| Rank | Feature | Group | MI (avg) | RF Perm Imp | PSI | Zero% | Recommendation |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---|")
    for i, r in enumerate(ranking.head(15).itertuples(index=False), 1):
        lines.append(
            f"| {i} | `{r.feature}` | {r.group} | "
            f"{_fmt(r.mi_reg_avg)} | {_fmt(r.perm_importance)} | "
            f"{_fmt(r.psi)} | {_fmt(r.zero_frac, 2)} | {r.recommendation} |"
        )
    lines.append("")

    lines.append("## Bottom 15 Features (Least Useful)")
    lines.append("")
    lines.append("| Rank | Feature | Group | MI (avg) | RF Perm Imp | PSI | Zero% | Recommendation |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---|")
    bottom = ranking.tail(15).iloc[::-1]
    for i, r in enumerate(bottom.itertuples(index=False), len(ranking) - 14):
        lines.append(
            f"| {i} | `{r.feature}` | {r.group} | "
            f"{_fmt(r.mi_reg_avg)} | {_fmt(r.perm_importance)} | "
            f"{_fmt(r.psi)} | {_fmt(r.zero_frac, 2)} | {r.recommendation} |"
        )
    lines.append("")

    if not corr_pairs.empty:
        lines.append("## Highly Correlated Feature Pairs")
        lines.append("")
        lines.append("| Feature A | Feature B | Pearson r | Tier |")
        lines.append("|---|---|---:|---|")
        for r in corr_pairs.head(20).itertuples(index=False):
            lines.append(f"| `{r.feature_a}` | `{r.feature_b}` | {_fmt(r.pearson_r)} | {r.tier} |")
        lines.append("")

    if uniqueness is not None and not uniqueness.empty:
        lines.append("## Feature Uniqueness (Correlated Pairs)")
        lines.append("")
        lines.append("| Feature A | Feature B | Corr | Score A Only | Score B Only | Score Both | Keep |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for r in uniqueness.itertuples(index=False):
            lines.append(
                f"| `{r.feature_a}` | `{r.feature_b}` | {_fmt(r.corr)} | "
                f"{_fmt(r.score_a_only)} | {_fmt(r.score_b_only)} | "
                f"{_fmt(r.score_both)} | `{r.recommend_keep}` |"
            )
        lines.append("")

    unstable = stability[stability["psi"] > 0.10] if not stability.empty else pd.DataFrame()
    if not unstable.empty:
        lines.append("## Unstable Features (PSI > 0.10)")
        lines.append("")
        lines.append("| Feature | PSI | KS Stat | Mean Shift (stds) |")
        lines.append("|---|---:|---:|---:|")
        for r in unstable.itertuples(index=False):
            lines.append(
                f"| `{r.feature}` | {_fmt(r.psi)} | {_fmt(r.ks_stat)} | {_fmt(r.mean_shift_stds)} |"
            )
        lines.append("")

    if cross_tf is not None and not cross_tf.empty:
        redundant_tf = cross_tf[cross_tf["redundant_flag"] == True]
        lines.append("## Cross-Timeframe Redundancy")
        lines.append(f"**{len(redundant_tf)}** features are redundant (|r| > 0.90) across timeframes.")
        lines.append("")
        lines.append("| Feature | r(5m,15m) | r(5m,45m) | r(15m,45m) | Redundant |")
        lines.append("|---|---:|---:|---:|---|")
        for r in cross_tf.itertuples(index=False):
            flag = "YES" if r.redundant_flag else ""
            lines.append(
                f"| `{r.feature}` | {_fmt(r.corr_5m_15m)} | "
                f"{_fmt(r.corr_5m_45m)} | {_fmt(r.corr_15m_45m)} | {flag} |"
            )
        lines.append("")

    if marginal is not None and not marginal.empty:
        lines.append("## Marginal Group Value")
        lines.append("")
        lines.append("| Group | N Features | Marginal Acc | Group Only Acc | Without Group Acc |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in marginal.itertuples(index=False):
            lines.append(
                f"| `{r.group}` | {r.n_features} | "
                f"{_fmt(r.marginal_accuracy)} | {_fmt(r.acc_group_only)} | "
                f"{_fmt(r.acc_without_group)} |"
            )
        lines.append("")

    lines.append("## Predictive Power Detail (MI Regression, all horizons)")
    lines.append("")
    pp_sorted = pred_power.sort_values("mi_reg_avg", ascending=False)
    lines.append("| Feature | MI 1h | MI 2h | MI 4h | MI 8h | MI Avg |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in pp_sorted.itertuples(index=False):
        vals = [_fmt(getattr(r, f"mi_reg_{h}")) for h in HORIZONS]
        lines.append(f"| `{r.feature}` | {' | '.join(vals)} | {_fmt(r.mi_reg_avg)} |")
    lines.append("")

    lines.append("## Sparse Features (>50% zeros in raw data)")
    lines.append("")
    sparse = stats[stats.get("zero_frac_raw", pd.Series(dtype=float)) > 0.50] if "zero_frac_raw" in stats.columns else pd.DataFrame()
    if not sparse.empty:
        sparse = sparse.sort_values("zero_frac_raw", ascending=False)
        lines.append("| Feature | Zero% (raw) | Unique Values |")
        lines.append("|---|---:|---:|")
        for r in sparse.itertuples(index=False):
            zf = getattr(r, "zero_frac_raw", 0)
            uq = getattr(r, "unique_raw", 0)
            lines.append(f"| `{r.feature}` | {_fmt(zf, 2)} | {uq} |")
    else:
        lines.append("None found.")
    lines.append("")

    report_path = output_dir / "FEATURE_ANALYSIS_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved to {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(ranking: pd.DataFrame, marginal: Optional[pd.DataFrame]) -> None:
    """Print concise console summary."""
    print("\n" + "=" * 70)
    print("  FEATURE IMPORTANCE ANALYSIS — US30 SUMMARY")
    print("=" * 70)

    print("\n  TOP 10 FEATURES (most useful for agent):")
    for i, r in enumerate(ranking.head(10).itertuples(index=False), 1):
        print(f"    {i:2d}. {r.feature:<30s}  MI={_fmt(r.mi_reg_avg)}  RF={_fmt(r.perm_importance)}  [{r.group}]")

    n_feats = len(ranking)
    print(f"\n  BOTTOM 10 FEATURES (least useful / noise):")
    for i, r in enumerate(ranking.tail(10).iloc[::-1].itertuples(index=False), n_feats - 9):
        print(f"    {i:2d}. {r.feature:<30s}  MI={_fmt(r.mi_reg_avg)}  RF={_fmt(r.perm_importance)}  [{r.recommendation}]")

    if marginal is not None and not marginal.empty:
        print("\n  FEATURE GROUP MARGINAL VALUE:")
        for r in marginal.itertuples(index=False):
            sign = "+" if r.marginal_accuracy >= 0 else ""
            print(f"    {r.group:<30s}  marginal={sign}{_fmt(r.marginal_accuracy)}  alone={_fmt(r.acc_group_only)}  ({r.n_features} feats)")

    keep = ranking[ranking["recommendation"].str.startswith("KEEP")]
    remove = ranking[~ranking["recommendation"].str.startswith("KEEP")]
    print(f"\n  RECOMMENDATION: Keep {len(keep)} features, review/remove {len(remove)} features")
    if not remove.empty:
        print("  Features to review:")
        for r in remove.itertuples(index=False):
            print(f"    - {r.feature:<30s}  {r.recommendation}")

    print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# CLI + Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature importance analysis for PPO agent (US30).")
    p.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "feature_analysis")
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--rf-trees", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-rf", action="store_true")
    p.add_argument("--skip-cross-tf", action="store_true")
    p.add_argument("--force-reengineer", action="store_true",
                   help="Force re-engineering features even if new features already exist")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    # 1. Load data (auto re-engineers if old features detected)
    data = load_data(args.processed_dir, force_reengineer=args.force_reengineer)
    cols = [c for c in MARKET_FEATURE_COLS if c in data["df_5m_norm"].columns]
    logger.info(f"Analyzing {len(cols)} features")

    # 2. Compute targets
    targets = compute_future_returns(data["df_5m_norm"]["close"], HORIZONS)

    _ensure_dir(args.output_dir)

    # 3. Statistical profiling
    stats = statistical_profile(data["df_5m_raw"], data["df_5m_norm"], cols)
    gc.collect()

    # 4. Correlation analysis
    corr_pairs, corr_matrix = correlation_analysis(
        data["df_5m_norm"], cols, data["train_mask"]
    )
    corr_matrix.to_csv(args.output_dir / "correlation_matrix.csv") if not corr_matrix.empty else None
    gc.collect()

    # 5. Predictive power
    pred_power = predictive_power(
        data["df_5m_norm"], targets, cols,
        data["train_mask"], args.sample_size, args.seed,
    )
    gc.collect()

    # 6. Temporal stability
    stability = temporal_stability(
        data["df_5m_norm"], cols, data["train_mask"], data["val_mask"],
    )
    gc.collect()

    # 7. Cross-timeframe redundancy
    cross_tf = None
    if not args.skip_cross_tf:
        cross_tf = cross_timeframe_redundancy(
            data["df_5m_norm"], data["df_15m_norm"], data["df_45m_norm"],
            cols, data["train_mask"], args.sample_size,
        )
        del data["df_15m_norm"], data["df_45m_norm"]
        gc.collect()

    # 8. RF importance
    rf_imp = None
    if not args.skip_rf:
        rf_imp = rf_importance(
            data["df_5m_norm"], targets, cols,
            data["train_mask"], args.sample_size, args.rf_trees, args.seed,
        )
        gc.collect()

    # 9. Feature uniqueness
    uniqueness = feature_uniqueness(
        data["df_5m_norm"], targets, cols, corr_pairs,
        data["train_mask"], args.sample_size, args.seed,
    )
    gc.collect()

    # 10. Marginal group value
    marginal = marginal_group_value(
        data["df_5m_norm"], targets, cols,
        data["train_mask"], args.sample_size, args.seed,
    )
    gc.collect()

    # 11. Composite ranking
    ranking = composite_ranking(pred_power, rf_imp, stability, stats, cols)

    # 12. Generate report
    _ensure_dir(args.output_dir)
    generate_report(
        stats, corr_pairs, pred_power, stability,
        cross_tf, rf_imp, uniqueness, marginal, ranking,
        args.output_dir,
    )

    # 13. Console summary
    print_summary(ranking, marginal)

    elapsed = time.time() - t0
    logger.info(f"Analysis complete in {elapsed:.1f}s. Results in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
