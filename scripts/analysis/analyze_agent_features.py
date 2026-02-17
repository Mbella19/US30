#!/usr/bin/env python3
"""
Comprehensive PPO Agent Feature Importance Analysis.

Performs rigorous testing of all 41 MARKET_FEATURE_COLS to identify:
1. Dead features (constants, zero information)
2. Low-importance features (policy ignores them)
3. Redundant features (highly correlated, keep best)
4. Feature groups that can be consolidated

Usage:
    python scripts/analysis/analyze_agent_features.py
    python scripts/analysis/analyze_agent_features.py --stats-only
    python scripts/analysis/analyze_agent_features.py --include-per-timeframe
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.live.bridge_constants import MARKET_FEATURE_COLS


# ============================================================================
# Constants
# ============================================================================

N_MARKET_FEATURES = len(MARKET_FEATURE_COLS)  # 41

# Observation vector layout (use_analyst=False, 145 dims)
OBS_BLOCKS = {
    "position_state": (0, 4),
    "market_5m": (4, 4 + N_MARKET_FEATURES),
    "market_15m": (4 + N_MARKET_FEATURES, 4 + 2 * N_MARKET_FEATURES),
    "market_45m": (4 + 2 * N_MARKET_FEATURES, 4 + 3 * N_MARKET_FEATURES),
    "sl_tp": (4 + 3 * N_MARKET_FEATURES, 4 + 3 * N_MARKET_FEATURES + 2),
    "hold_features": (4 + 3 * N_MARKET_FEATURES + 2, 4 + 3 * N_MARKET_FEATURES + 6),
    "returns_window": (4 + 3 * N_MARKET_FEATURES + 6, 4 + 3 * N_MARKET_FEATURES + 18),
}

# Pre-defined redundancy groups to test
REDUNDANCY_GROUPS = {
    "dead_v36": ["volatility_regime", "distribution_shift_score"],
    "mean_reversion_oscillators": ["rsi", "williams_r", "bb_percent_b", "price_zscore", "cci"],
    "mean_reversion_other": ["bb_bandwidth", "rsi_divergence"],
    "regime_indicators": ["chop", "adx"],
    "volatility_representations": ["atr", "atr_context", "atr_percentile"],
    "sma_representations": ["sma_distance", "sma_distance_percentile"],
    "ood_v37": ["volatility_vs_training", "returns_skew_shift", "atr_vs_training", "range_vs_training", "ood_score"],
    "structure_breaks": ["structure_fade", "bars_since_bos", "bars_since_choch", "bos_magnitude", "choch_magnitude"],
    "sr_strength": ["sr_strength_r", "sr_strength_s"],
    "sessions": ["session_asian", "session_london", "session_ny"],
    "percentiles": ["atr_percentile", "chop_percentile", "sma_distance_percentile", "volatility_percentile"],
}


# ============================================================================
# Part 1: Statistical Feature Analysis
# ============================================================================

def compute_feature_stats(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Per-feature statistics: mean, std, min, max, zero_frac, unique, entropy."""
    cols = [c for c in columns if c in df.columns]
    rows = []
    for col in cols:
        vals = df[col].to_numpy()
        non_na = vals[~np.isnan(vals)]
        n = len(vals)
        n_valid = len(non_na)

        zero_frac = float((non_na == 0).mean()) if n_valid else 0.0
        nan_frac = float(np.isnan(vals).mean()) if n else 0.0
        unique_count = int(pd.Series(non_na).nunique()) if n_valid else 0
        clip_frac = float((np.abs(non_na) >= 4.999).mean()) if n_valid else 0.0

        # Shannon entropy (bin continuous into 50 quantile buckets)
        entropy = 0.0
        if n_valid > 0 and unique_count > 1:
            if unique_count <= 10:
                # Discrete feature
                _, counts = np.unique(non_na, return_counts=True)
                p = counts / counts.sum()
                entropy = float(-np.sum(p * np.log2(p + 1e-12)))
            else:
                # Continuous: quantile binning
                try:
                    bins = np.percentile(non_na, np.linspace(0, 100, 51))
                    bins = np.unique(bins)
                    if len(bins) > 1:
                        counts, _ = np.histogram(non_na, bins=bins)
                        p = counts / counts.sum()
                        p = p[p > 0]
                        entropy = float(-np.sum(p * np.log2(p)))
                except Exception:
                    entropy = 0.0

        rows.append({
            "feature": col,
            "count": n,
            "nan_frac": nan_frac,
            "zero_frac": zero_frac,
            "unique": unique_count,
            "mean": float(np.mean(non_na)) if n_valid else np.nan,
            "std": float(np.std(non_na)) if n_valid else np.nan,
            "min": float(np.min(non_na)) if n_valid else np.nan,
            "max": float(np.max(non_na)) if n_valid else np.nan,
            "clip_frac": clip_frac,
            "entropy": entropy,
        })

    return pd.DataFrame(rows)


def find_dead_features(df: pd.DataFrame, columns: List[str], std_threshold: float = 1e-6) -> List[str]:
    """Find features with near-zero variance (constants)."""
    dead = []
    for col in columns:
        if col in df.columns:
            std = df[col].std()
            if std < std_threshold or df[col].nunique() <= 1:
                dead.append(col)
    return dead


def compute_correlation_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Full Pearson correlation matrix."""
    cols = [c for c in columns if c in df.columns]
    return df[cols].corr(method="pearson")


def find_high_corr_pairs(corr: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Extract pairs with |r| >= threshold."""
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = []
    abs_vals = np.abs(upper.values)
    hits = np.argwhere(abs_vals >= threshold)
    for i, j in hits.tolist():
        fa = str(upper.index[i])
        fb = str(upper.columns[j])
        c = float(upper.iat[i, j])
        pairs.append({"feature_a": fa, "feature_b": fb, "corr": c, "abs_corr": abs(c)})
    if not pairs:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr", "abs_corr"])
    return pd.DataFrame(pairs).sort_values("abs_corr", ascending=False)


# ============================================================================
# Part 2: Agent Policy Sensitivity Analysis
# ============================================================================

def build_analysis_env(processed_dir: Path, config):
    """Build TradingEnv matching training config for sensitivity analysis."""
    from src.live.bridge_constants import MODEL_FEATURE_COLS
    from src.training.train_agent import prepare_env_data, create_trading_env

    feature_cols = list(MODEL_FEATURE_COLS)

    # Load normalized parquets
    df_5m = pd.read_parquet(processed_dir / "features_5m_normalized.parquet")
    df_15m = pd.read_parquet(processed_dir / "features_15m_normalized.parquet")
    df_45m = pd.read_parquet(processed_dir / "features_45m_normalized.parquet")

    # Prepare data
    data_5m, data_15m, data_45m, close_prices, market_features, returns, rolling_lookback_data = prepare_env_data(
        df_5m, df_15m, df_45m, feature_cols,
        config.analyst.lookback_5m,
        config.analyst.lookback_15m,
        config.analyst.lookback_45m,
    )

    # Split: use validation portion (70-85%) for analysis
    n_total = len(close_prices)
    train_end = int(0.70 * n_total)
    val_end = int(0.85 * n_total)

    # Compute stats from training data
    train_mkt = market_features[:train_end]
    mkt_mean = train_mkt.mean(axis=0).astype(np.float32)
    mkt_std = train_mkt.std(axis=0).astype(np.float32)
    mkt_std = np.where(mkt_std > 1e-8, mkt_std, 1.0).astype(np.float32)

    # Rolling warmup from training tail
    rolling_ws = config.normalization.rolling_window_size
    warmup_start = max(0, train_end - rolling_ws)
    rolling_warmup = market_features[warmup_start:train_end].astype(np.float32)

    # OHLC for env
    ohlc = None
    if all(c in df_5m.columns for c in ["open", "high", "low", "close"]):
        start_idx = max(
            config.analyst.lookback_5m,
            (config.analyst.lookback_15m - 1) * 3 + 1,
            (config.analyst.lookback_45m - 1) * 9 + 1,
        )
        n_samples = len(df_5m) - start_idx
        ohlc = df_5m[["open", "high", "low", "close"]].values[start_idx:start_idx + n_samples].astype(np.float32)

    # Use validation slice for analysis
    env = create_trading_env(
        data_5m=data_5m[train_end:val_end],
        data_15m=data_15m[train_end:val_end],
        data_45m=data_45m[train_end:val_end],
        close_prices=close_prices[train_end:val_end],
        market_features=market_features[train_end:val_end],
        returns=returns[train_end:val_end] if returns is not None else None,
        analyst_model=None,
        config=config,
        device=torch.device("cpu"),
        market_feat_mean=mkt_mean,
        market_feat_std=mkt_std,
        use_analyst=False,
        use_regime_sampling=False,
        ohlc_data=ohlc[train_end:val_end] if ohlc is not None else None,
        rolling_lookback_data=rolling_warmup,
    )
    return env


def collect_observations(env, agent, n_samples: int = 10000, seed: int = 42) -> np.ndarray:
    """Collect realistic observations by running the trained policy."""
    obs_list = []
    obs, _ = env.reset(seed=seed)
    for _ in range(n_samples):
        obs_list.append(np.asarray(obs, dtype=np.float32).copy())
        action, _ = agent.model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            obs, _ = env.reset()
    return np.stack(obs_list)


def _action_change_stats(
    base_actions: np.ndarray, new_actions: np.ndarray
) -> Tuple[float, float, float]:
    """Compute fraction of changed actions."""
    any_changed = float((new_actions != base_actions).any(axis=1).mean())
    dir_changed = float((new_actions[:, 0] != base_actions[:, 0]).mean())
    size_changed = float((new_actions[:, 1] != base_actions[:, 1]).mean())
    return any_changed, dir_changed, size_changed


def _permute_cols(
    obs_batch: np.ndarray, agent, base_actions: np.ndarray,
    col_indices: Sequence[int], rng: np.random.Generator
) -> Tuple[float, float, float]:
    """Permute specified columns and measure action changes."""
    perm = rng.permutation(obs_batch.shape[0])
    x = obs_batch.copy()
    cols = np.array(col_indices, dtype=np.int64)
    x[:, cols] = x[perm][:, cols]
    acts, _ = agent.model.predict(x, deterministic=True)
    return _action_change_stats(base_actions, np.asarray(acts))


def block_permutation_importance(
    obs_batch: np.ndarray, agent, n_repeats: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Block-level permutation importance."""
    rng = np.random.default_rng(seed)
    base_actions, _ = agent.model.predict(obs_batch, deterministic=True)
    base_actions = np.asarray(base_actions)

    rows = []
    for block_name, (start, end) in OBS_BLOCKS.items():
        any_list, dir_list, size_list = [], [], []
        for _ in range(n_repeats):
            a, d, s = _permute_cols(obs_batch, agent, base_actions, list(range(start, end)), rng)
            any_list.append(a)
            dir_list.append(d)
            size_list.append(s)
        rows.append({
            "block": block_name,
            "dims": end - start,
            "any_changed_mean": float(np.mean(any_list)),
            "any_changed_std": float(np.std(any_list)),
            "direction_changed_mean": float(np.mean(dir_list)),
            "size_changed_mean": float(np.mean(size_list)),
        })

    return pd.DataFrame(rows).sort_values("any_changed_mean", ascending=False)


def feature_permutation_importance(
    obs_batch: np.ndarray, agent, n_repeats: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Per-feature permutation across all 3 timeframes simultaneously."""
    rng = np.random.default_rng(seed)
    base_actions, _ = agent.model.predict(obs_batch, deterministic=True)
    base_actions = np.asarray(base_actions)

    rows = []
    for i, feat_name in enumerate(MARKET_FEATURE_COLS):
        # Same feature across all 3 timeframes
        cols = [4 + i, 4 + N_MARKET_FEATURES + i, 4 + 2 * N_MARKET_FEATURES + i]
        any_list, dir_list, size_list = [], [], []
        for _ in range(n_repeats):
            a, d, s = _permute_cols(obs_batch, agent, base_actions, cols, rng)
            any_list.append(a)
            dir_list.append(d)
            size_list.append(s)
        rows.append({
            "feature": feat_name,
            "any_changed_mean": float(np.mean(any_list)),
            "any_changed_std": float(np.std(any_list)),
            "direction_changed_mean": float(np.mean(dir_list)),
            "size_changed_mean": float(np.mean(size_list)),
        })

    return pd.DataFrame(rows).sort_values("any_changed_mean", ascending=False)


def cross_timeframe_importance(
    obs_batch: np.ndarray, agent, n_repeats: int = 3, seed: int = 42
) -> pd.DataFrame:
    """Per-feature, per-timeframe permutation importance."""
    rng = np.random.default_rng(seed)
    base_actions, _ = agent.model.predict(obs_batch, deterministic=True)
    base_actions = np.asarray(base_actions)

    tf_offsets = {"5m": 4, "15m": 4 + N_MARKET_FEATURES, "45m": 4 + 2 * N_MARKET_FEATURES}
    rows = []

    for i, feat_name in enumerate(MARKET_FEATURE_COLS):
        for tf_name, offset in tf_offsets.items():
            cols = [offset + i]
            any_list = []
            for _ in range(n_repeats):
                a, _, _ = _permute_cols(obs_batch, agent, base_actions, cols, rng)
                any_list.append(a)
            rows.append({
                "feature": feat_name,
                "timeframe": tf_name,
                "any_changed_mean": float(np.mean(any_list)),
                "any_changed_std": float(np.std(any_list)),
            })

    return pd.DataFrame(rows)


# ============================================================================
# Part 3: Feature Redundancy Analysis
# ============================================================================

def hierarchical_cluster_features(corr_matrix: pd.DataFrame, threshold: float = 0.3):
    """Cluster features using (1 - |corr|) as distance."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    distance = 1 - np.abs(corr_matrix.values)
    np.fill_diagonal(distance, 0)
    # Replace NaN/inf with max distance (uncorrelated)
    distance = np.nan_to_num(distance, nan=1.0, posinf=1.0, neginf=0.0)
    # Ensure symmetry and valid range
    distance = np.clip((distance + distance.T) / 2, 0, 1)
    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=threshold, criterion="distance")

    cluster_df = pd.DataFrame({
        "feature": corr_matrix.index.tolist(),
        "cluster": clusters.tolist(),
    }).sort_values("cluster")

    return cluster_df, Z


def analyze_redundancy_groups(
    corr_matrix: pd.DataFrame,
    importance_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Analyze pre-defined redundancy groups."""
    rows = []
    for group_name, features in REDUNDANCY_GROUPS.items():
        valid = [f for f in features if f in corr_matrix.index]
        if len(valid) < 2:
            continue

        # Mean intra-group correlation
        sub_corr = corr_matrix.loc[valid, valid]
        upper = sub_corr.where(np.triu(np.ones(sub_corr.shape), k=1).astype(bool))
        mean_corr = float(np.abs(upper.values[~np.isnan(upper.values)]).mean())

        # Get importance scores if available
        best_feature = None
        best_importance = 0.0
        worst_feature = None
        worst_importance = 1.0
        if importance_df is not None:
            for f in valid:
                imp = importance_df.loc[importance_df["feature"] == f, "any_changed_mean"]
                if not imp.empty:
                    val = float(imp.iloc[0])
                    if val > best_importance:
                        best_importance = val
                        best_feature = f
                    if val < worst_importance:
                        worst_importance = val
                        worst_feature = f

        rows.append({
            "group": group_name,
            "features": ", ".join(valid),
            "n_features": len(valid),
            "mean_abs_corr": mean_corr,
            "best_feature": best_feature or "N/A",
            "best_importance": best_importance,
            "worst_feature": worst_feature or "N/A",
            "worst_importance": worst_importance,
        })

    return pd.DataFrame(rows).sort_values("mean_abs_corr", ascending=False)


# ============================================================================
# Visualization
# ============================================================================

def save_correlation_heatmap(corr: pd.DataFrame, out_path: Path):
    """Save correlation matrix as heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(18, 15))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(corr.index, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Feature Correlation Matrix (41 MARKET_FEATURE_COLS)", fontsize=12)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  Warning: Could not save heatmap: {e}")


def save_dendrogram(Z, labels: List[str], out_path: Path):
    """Save feature dendrogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram

        fig, ax = plt.subplots(figsize=(16, 8))
        dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=7,
                    color_threshold=0.3)
        ax.set_title("Feature Hierarchical Clustering (distance = 1 - |corr|)", fontsize=12)
        ax.set_ylabel("Distance (1 - |correlation|)")
        ax.axhline(y=0.3, color="r", linestyle="--", alpha=0.5, label="Threshold (|corr|=0.7)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  Warning: Could not save dendrogram: {e}")


def save_importance_bar_chart(importance_df: pd.DataFrame, out_path: Path):
    """Save feature importance bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = importance_df.sort_values("any_changed_mean", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 14))
        colors = ["#d32f2f" if v < 0.01 else "#ff9800" if v < 0.03 else "#4caf50"
                   for v in df["any_changed_mean"]]
        ax.barh(range(len(df)), df["any_changed_mean"], color=colors, height=0.7)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["feature"], fontsize=8)
        ax.set_xlabel("Action Change Rate (permutation importance)")
        ax.set_title("PPO Agent Feature Sensitivity\n(red = low, orange = medium, green = high)")
        ax.axvline(x=0.01, color="red", linestyle="--", alpha=0.5, label="Low threshold (1%)")
        ax.axvline(x=0.03, color="orange", linestyle="--", alpha=0.5, label="Medium threshold (3%)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  Warning: Could not save bar chart: {e}")


# ============================================================================
# Report Generation
# ============================================================================

def _fmt(x, digits=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "N/A"
    return f"{x:.{digits}f}"


def generate_report(
    out_path: Path,
    dead_features: List[str],
    stats_raw: pd.DataFrame,
    stats_norm: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    block_sens: Optional[pd.DataFrame],
    feat_sens: Optional[pd.DataFrame],
    tf_sens: Optional[pd.DataFrame],
    redundancy_groups: Optional[pd.DataFrame],
    cluster_df: Optional[pd.DataFrame],
):
    """Generate comprehensive markdown report."""
    lines = []
    lines.append("# PPO Agent Feature Importance Report")
    lines.append("")
    lines.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Features analyzed**: {N_MARKET_FEATURES} MARKET_FEATURE_COLS x 3 timeframes = {N_MARKET_FEATURES * 3} market dims")
    lines.append(f"**Total observation dims**: 145 (4 pos + {N_MARKET_FEATURES * 3} market + 2 SL/TP + 4 hold + 12 returns)")
    lines.append("")

    # Section 1: Dead Features
    lines.append("## 1. Dead Features (IMMEDIATE DROP - zero information)")
    lines.append("")
    if dead_features:
        lines.append(f"**{len(dead_features)} dead features found** (constant value, zero variance):")
        lines.append("")
        for f in dead_features:
            row = stats_norm[stats_norm["feature"] == f]
            if not row.empty:
                r = row.iloc[0]
                lines.append(f"- `{f}`: mean={_fmt(r['mean'])}, std={_fmt(r['std'])}, unique={int(r['unique'])}")
        lines.append("")
        lines.append(f"**Impact**: Dropping these saves **{len(dead_features)} x 3 TFs = {len(dead_features) * 3} dims** from observation vector.")
    else:
        lines.append("No dead features found.")
    lines.append("")

    # Section 2: Block Sensitivity
    if block_sens is not None:
        lines.append("## 2. Observation Block Sensitivity")
        lines.append("")
        lines.append("Shows which blocks of the 145-dim observation the policy actually uses.")
        lines.append("Higher action change rate = more important to the policy.")
        lines.append("")
        lines.append("| Rank | Block | Dims | Action Change Rate | Direction Change | Size Change |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for i, row in enumerate(block_sens.itertuples(index=False), 1):
            lines.append(
                f"| {i} | `{row.block}` | {row.dims} | "
                f"{_fmt(row.any_changed_mean, 3)} ± {_fmt(row.any_changed_std, 3)} | "
                f"{_fmt(row.direction_changed_mean, 3)} | {_fmt(row.size_changed_mean, 3)} |"
            )
        lines.append("")

    # Section 3: Feature Importance
    if feat_sens is not None:
        lines.append("## 3. Feature Importance Ranking (all 3 timeframes)")
        lines.append("")
        lines.append("Permutation importance: shuffle each feature across all 3 TFs, measure action changes.")
        lines.append("")

        sorted_fs = feat_sens.sort_values("any_changed_mean", ascending=False)

        # Top features
        lines.append("### Top 15 Most Important Features")
        lines.append("")
        lines.append("| Rank | Feature | Action Change | Direction Change | Size Change |")
        lines.append("|---:|---|---:|---:|---:|")
        for i, row in enumerate(sorted_fs.head(15).itertuples(index=False), 1):
            lines.append(
                f"| {i} | `{row.feature}` | "
                f"{_fmt(row.any_changed_mean, 4)} ± {_fmt(row.any_changed_std, 4)} | "
                f"{_fmt(row.direction_changed_mean, 4)} | {_fmt(row.size_changed_mean, 4)} |"
            )
        lines.append("")

        # Bottom features (drop candidates)
        lines.append("### Bottom 15 Least Important Features (DROP CANDIDATES)")
        lines.append("")
        lines.append("| Rank | Feature | Action Change | Direction Change | Size Change |")
        lines.append("|---:|---|---:|---:|---:|")
        bottom = sorted_fs.tail(15).sort_values("any_changed_mean", ascending=True)
        for i, row in enumerate(bottom.itertuples(index=False), 1):
            marker = " **DEAD**" if row.feature in dead_features else ""
            lines.append(
                f"| {i} | `{row.feature}`{marker} | "
                f"{_fmt(row.any_changed_mean, 4)} ± {_fmt(row.any_changed_std, 4)} | "
                f"{_fmt(row.direction_changed_mean, 4)} | {_fmt(row.size_changed_mean, 4)} |"
            )
        lines.append("")

    # Section 4: Timeframe Importance
    if tf_sens is not None:
        lines.append("## 4. Timeframe Importance per Feature")
        lines.append("")
        lines.append("Shows which timeframe the policy relies on most for each feature.")
        lines.append("")
        lines.append("| Feature | 5m Change | 15m Change | 45m Change | Primary TF |")
        lines.append("|---|---:|---:|---:|---|")
        features = tf_sens["feature"].unique()
        for feat in features:
            sub = tf_sens[tf_sens["feature"] == feat]
            vals = {}
            for _, row in sub.iterrows():
                vals[row["timeframe"]] = row["any_changed_mean"]
            primary = max(vals, key=vals.get) if vals else "N/A"
            lines.append(
                f"| `{feat}` | {_fmt(vals.get('5m', 0), 4)} | "
                f"{_fmt(vals.get('15m', 0), 4)} | {_fmt(vals.get('45m', 0), 4)} | **{primary}** |"
            )
        lines.append("")

    # Section 5: Correlation / Redundancy
    lines.append("## 5. High-Correlation Pairs (|r| >= 0.7)")
    lines.append("")
    if not corr_pairs.empty:
        lines.append(f"Found **{len(corr_pairs)} pairs** with |correlation| >= 0.7:")
        lines.append("")
        lines.append("| Feature A | Feature B | Correlation |")
        lines.append("|---|---|---:|")
        for row in corr_pairs.head(25).itertuples(index=False):
            lines.append(f"| `{row.feature_a}` | `{row.feature_b}` | {_fmt(row.corr, 3)} |")
        lines.append("")
    else:
        lines.append("No pairs with |correlation| >= 0.7 found.")
        lines.append("")

    # Section 6: Redundancy Groups
    if redundancy_groups is not None and not redundancy_groups.empty:
        lines.append("## 6. Redundancy Group Analysis")
        lines.append("")
        lines.append("| Group | N Features | Mean |corr| | Best Feature | Best Imp. | Worst Feature | Worst Imp. |")
        lines.append("|---|---:|---:|---|---:|---|---:|")
        for row in redundancy_groups.itertuples(index=False):
            lines.append(
                f"| {row.group} | {row.n_features} | {_fmt(row.mean_abs_corr, 3)} | "
                f"`{row.best_feature}` | {_fmt(row.best_importance, 4)} | "
                f"`{row.worst_feature}` | {_fmt(row.worst_importance, 4)} |"
            )
        lines.append("")

        # Detailed group breakdown
        for _, row in redundancy_groups.iterrows():
            lines.append(f"### Group: {row['group']}")
            features = [f.strip() for f in row["features"].split(",")]
            lines.append(f"Features: {', '.join(f'`{f}`' for f in features)}")
            lines.append(f"Mean intra-group |correlation|: **{_fmt(row['mean_abs_corr'], 3)}**")
            if row["best_feature"] != "N/A":
                lines.append(f"**KEEP**: `{row['best_feature']}` (importance: {_fmt(row['best_importance'], 4)})")
                drop_candidates = [f for f in features if f != row["best_feature"]]
                if drop_candidates:
                    lines.append(f"Consider dropping: {', '.join(f'`{f}`' for f in drop_candidates)}")
            lines.append("")

    # Section 7: Sparsity Analysis
    lines.append("## 7. Feature Sparsity Analysis")
    lines.append("")
    sparse_features = stats_raw[stats_raw["zero_frac"] > 0.8].sort_values("zero_frac", ascending=False)
    if not sparse_features.empty:
        lines.append(f"**{len(sparse_features)} features** with >80% zeros:")
        lines.append("")
        lines.append("| Feature | Zero % | Unique Values | Entropy | Importance |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in sparse_features.iterrows():
            imp = "N/A"
            if feat_sens is not None:
                imp_row = feat_sens[feat_sens["feature"] == row["feature"]]
                if not imp_row.empty:
                    imp = _fmt(float(imp_row.iloc[0]["any_changed_mean"]), 4)
            lines.append(
                f"| `{row['feature']}` | {_fmt(row['zero_frac'] * 100, 1)}% | "
                f"{int(row['unique'])} | {_fmt(row['entropy'], 2)} | {imp} |"
            )
        lines.append("")
    else:
        lines.append("No features with >80% zeros.")
        lines.append("")

    # Section 8: Final Recommendations
    lines.append("## 8. Final Recommendations")
    lines.append("")

    # Categorize features
    drop_safe = list(dead_features)
    drop_low = []
    keep = []
    merge_candidates = []

    if feat_sens is not None:
        for _, row in feat_sens.iterrows():
            f = row["feature"]
            imp = row["any_changed_mean"]
            if f in dead_features:
                continue
            if imp < 0.005:
                drop_low.append((f, imp))
            elif imp >= 0.02:
                keep.append((f, imp))
            else:
                merge_candidates.append((f, imp))

    lines.append("### DROP (safe - zero information):")
    if drop_safe:
        for f in drop_safe:
            lines.append(f"- `{f}` — constant value, zero variance")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("### DROP (low value - policy ignores):")
    if drop_low:
        for f, imp in sorted(drop_low, key=lambda x: x[1]):
            lines.append(f"- `{f}` — action change rate: {_fmt(imp, 4)}")
    else:
        lines.append("- None definitively identified (run with more samples for precision)")
    lines.append("")

    lines.append("### KEEP (confirmed important):")
    if keep:
        for f, imp in sorted(keep, key=lambda x: -x[1]):
            lines.append(f"- `{f}` — action change rate: {_fmt(imp, 4)}")
    else:
        lines.append("- Analysis pending (run with agent checkpoint)")
    lines.append("")

    lines.append("### REVIEW (moderate importance, possible redundancy):")
    if merge_candidates:
        for f, imp in sorted(merge_candidates, key=lambda x: x[1]):
            lines.append(f"- `{f}` — action change rate: {_fmt(imp, 4)}")
    else:
        lines.append("- None")
    lines.append("")

    # Summary
    n_drop = len(drop_safe) + len(drop_low)
    n_keep = len(keep)
    n_review = len(merge_candidates)
    lines.append("### Summary")
    lines.append(f"- **Current**: 41 features x 3 TFs = 123 market dims + 22 other = 145 total")
    if n_drop > 0:
        new_market = (N_MARKET_FEATURES - n_drop) * 3
        new_total = 4 + new_market + 2 + 4 + 12
        lines.append(f"- **Proposed**: {N_MARKET_FEATURES - n_drop} features x 3 TFs = {new_market} market dims + 22 other = {new_total} total")
        lines.append(f"- **Reduction**: {n_drop} features removed = {n_drop * 3} dims saved ({n_drop * 3 / 123 * 100:.0f}% of market dims)")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {out_path}")


# ============================================================================
# Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO Agent Feature Importance Analysis")
    parser.add_argument("--processed-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="PPO checkpoint path. Auto-detects latest if not set.")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "agent_feature_audit")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--corr-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-only", action="store_true",
                        help="Only run statistical analysis (no model needed)")
    parser.add_argument("--include-per-timeframe", action="store_true",
                        help="Include per-timeframe importance (slower)")
    return parser.parse_args()


def find_latest_checkpoint(agent_dir: Path) -> Optional[Path]:
    """Find the latest PPO checkpoint."""
    ckpt_dir = agent_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("sniper_model_*_steps.zip"))
    if not checkpoints:
        return None
    return checkpoints[-1]


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from config.settings import Config
    config = Config()

    print("=" * 70)
    print("PPO Agent Feature Importance Analysis")
    print("=" * 70)
    print(f"Features: {N_MARKET_FEATURES} MARKET_FEATURE_COLS x 3 TFs")
    print(f"Output: {out_dir}")
    print()

    # ----------------------------------------------------------------
    # Part 1: Statistical Analysis
    # ----------------------------------------------------------------
    print("=" * 50)
    print("PART 1: Statistical Feature Analysis")
    print("=" * 50)

    # Load raw and normalized 5m data
    raw_cols = [c for c in MARKET_FEATURE_COLS]
    extra = ["open", "high", "low", "close"]
    load_cols = sorted(set(raw_cols + extra))

    df_raw = pd.read_parquet(args.processed_dir / "features_5m.parquet")
    df_norm = pd.read_parquet(args.processed_dir / "features_5m_normalized.parquet")

    print(f"  Loaded {len(df_raw):,} rows of raw data")
    print(f"  Loaded {len(df_norm):,} rows of normalized data")

    # Dead features
    dead = find_dead_features(df_norm, raw_cols)
    print(f"\n  Dead features (zero variance): {dead if dead else 'None'}")

    # Feature stats
    stats_raw = compute_feature_stats(df_raw, raw_cols)
    stats_norm = compute_feature_stats(df_norm, raw_cols)
    stats_raw.to_csv(out_dir / "feature_stats_raw.csv", index=False)
    stats_norm.to_csv(out_dir / "feature_stats_normalized.csv", index=False)
    print(f"  Saved feature stats CSVs")

    # Sparsity
    sparse = stats_raw[stats_raw["zero_frac"] > 0.8]
    if not sparse.empty:
        print(f"\n  Sparse features (>80% zeros):")
        for _, row in sparse.sort_values("zero_frac", ascending=False).iterrows():
            print(f"    {row['feature']:>30s}: {row['zero_frac']*100:.1f}% zeros, {int(row['unique'])} unique")

    # Correlation
    corr_matrix = compute_correlation_matrix(df_norm, raw_cols)
    corr_matrix.to_csv(out_dir / "correlation_matrix.csv")
    save_correlation_heatmap(corr_matrix, out_dir / "correlation_heatmap.png")

    corr_pairs = find_high_corr_pairs(corr_matrix, args.corr_threshold)
    corr_pairs.to_csv(out_dir / "high_corr_pairs.csv", index=False)
    print(f"\n  High-correlation pairs (|r| >= {args.corr_threshold}): {len(corr_pairs)}")
    for _, row in corr_pairs.head(10).iterrows():
        print(f"    {row['feature_a']:>25s} <-> {row['feature_b']:<25s}  r={row['corr']:.3f}")

    # Hierarchical clustering
    cluster_df, Z = hierarchical_cluster_features(corr_matrix, threshold=0.3)
    cluster_df.to_csv(out_dir / "redundancy_clusters.csv", index=False)
    save_dendrogram(Z, corr_matrix.columns.tolist(), out_dir / "dendrogram.png")

    n_clusters = cluster_df["cluster"].nunique()
    print(f"\n  Hierarchical clusters (distance < 0.3 = |corr| > 0.7): {n_clusters} clusters")
    for c in sorted(cluster_df["cluster"].unique()):
        members = cluster_df[cluster_df["cluster"] == c]["feature"].tolist()
        if len(members) > 1:
            print(f"    Cluster {c}: {', '.join(members)}")

    if args.stats_only:
        print("\n--stats-only mode: skipping agent sensitivity analysis")
        redundancy_groups = analyze_redundancy_groups(corr_matrix, None)
        redundancy_groups.to_csv(out_dir / "redundancy_groups.csv", index=False)
        generate_report(
            out_dir / "REPORT.md", dead, stats_raw, stats_norm, corr_pairs,
            None, None, None, redundancy_groups, cluster_df
        )
        return 0

    # ----------------------------------------------------------------
    # Part 2: Agent Policy Sensitivity
    # ----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PART 2: Agent Policy Sensitivity Analysis")
    print("=" * 50)

    # Find checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(PROJECT_ROOT / "models" / "agent")
    if ckpt_path is None or not ckpt_path.exists():
        print("  ERROR: No agent checkpoint found. Run with --stats-only or specify --checkpoint")
        return 1

    print(f"  Checkpoint: {ckpt_path}")

    # Build env
    print("  Building analysis environment...")
    env = build_analysis_env(args.processed_dir, config)
    print(f"  Observation space: {env.observation_space.shape}")

    # Load agent
    print("  Loading PPO agent...")
    from src.agents.sniper_agent import SniperAgent
    agent = SniperAgent.load(str(ckpt_path), env, device="cpu")
    print("  Agent loaded successfully")

    # Collect observations
    print(f"  Collecting {args.n_samples} observations...")
    obs_batch = collect_observations(env, agent, n_samples=args.n_samples, seed=args.seed)
    print(f"  Observations shape: {obs_batch.shape}")

    # Block-level sensitivity
    print("\n  Computing block-level sensitivity...")
    block_sens = block_permutation_importance(obs_batch, agent, n_repeats=args.n_repeats, seed=args.seed)
    block_sens.to_csv(out_dir / "block_sensitivity.csv", index=False)
    print("  Block sensitivity results:")
    for row in block_sens.itertuples(index=False):
        bar = "█" * int(row.any_changed_mean * 100)
        print(f"    {row.block:>20s} ({row.dims:>3d} dims): {row.any_changed_mean:.4f} {bar}")

    # Feature-level sensitivity
    print(f"\n  Computing per-feature sensitivity ({N_MARKET_FEATURES} features x {args.n_repeats} repeats)...")
    feat_sens = feature_permutation_importance(obs_batch, agent, n_repeats=args.n_repeats, seed=args.seed)
    feat_sens.to_csv(out_dir / "feature_sensitivity_all_tf.csv", index=False)
    save_importance_bar_chart(feat_sens, out_dir / "feature_importance.png")

    print("\n  Top 10 most important features:")
    for i, row in enumerate(feat_sens.head(10).itertuples(index=False), 1):
        print(f"    {i:>2d}. {row.feature:>30s}: {row.any_changed_mean:.4f}")

    print("\n  Bottom 10 least important features:")
    bottom = feat_sens.tail(10).sort_values("any_changed_mean", ascending=True)
    for i, row in enumerate(bottom.itertuples(index=False), 1):
        dead_marker = " [DEAD]" if row.feature in dead else ""
        print(f"    {i:>2d}. {row.feature:>30s}: {row.any_changed_mean:.4f}{dead_marker}")

    # Per-timeframe sensitivity (optional)
    tf_sens = None
    if args.include_per_timeframe:
        print(f"\n  Computing per-timeframe sensitivity ({N_MARKET_FEATURES} x 3 TFs x {args.n_repeats} repeats)...")
        tf_sens = cross_timeframe_importance(obs_batch, agent, n_repeats=args.n_repeats, seed=args.seed)
        tf_sens.to_csv(out_dir / "feature_sensitivity_per_tf.csv", index=False)
        print("  Per-timeframe results saved")

    # Cleanup env
    env.close()
    gc.collect()

    # ----------------------------------------------------------------
    # Part 3: Redundancy Analysis
    # ----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PART 3: Feature Redundancy Analysis")
    print("=" * 50)

    redundancy_groups = analyze_redundancy_groups(corr_matrix, feat_sens)
    redundancy_groups.to_csv(out_dir / "redundancy_groups.csv", index=False)

    print("\n  Redundancy groups:")
    for _, row in redundancy_groups.iterrows():
        print(f"    {row['group']:>30s}: mean|corr|={row['mean_abs_corr']:.3f}, "
              f"best={row['best_feature']} ({row['best_importance']:.4f})")

    # ----------------------------------------------------------------
    # Part 4: Generate Report
    # ----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PART 4: Generating Report")
    print("=" * 50)

    generate_report(
        out_dir / "REPORT.md", dead, stats_raw, stats_norm, corr_pairs,
        block_sens, feat_sens, tf_sens, redundancy_groups, cluster_df
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"All outputs saved to: {out_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
