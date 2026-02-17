#!/usr/bin/env python3
"""
Run an out-of-sample (OOS) backtest on a user-provided 1-minute OHLC CSV.

This script:
1) Loads the raw CSV (datetime, open, high, low, close)
2) Resamples to 5m/15m/45m, engineers features, aligns timeframes
3) Applies the EXISTING feature normalizers from models/analyst/normalizer_*.pkl
4) Loads the EXISTING Analyst checkpoint (models/analyst/best.pt)
5) Loads a provided Sniper Agent checkpoint (.zip) and runs a full backtest

Outputs are written to results/<timestamp>_oos_csv/.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Ensure project root is on the import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config, get_device  # noqa: E402
from src.agents.sniper_agent import SniperAgent  # noqa: E402
from src.data.features import engineer_all_features  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.normalizer import FeatureNormalizer  # noqa: E402
from src.data.resampler import align_timeframes, resample_all_timeframes  # noqa: E402
from src.evaluation.backtest import (  # noqa: E402
    compare_with_baseline,
    print_comparison_report,
    run_backtest,
    save_backtest_results,
)
from src.evaluation.ood_detector import analyze_distribution_shift  # noqa: E402
from src.data.ood_features import TrainingBaseline  # noqa: E402
from src.evaluation.metrics import print_metrics_report  # noqa: E402
from src.live.bridge_constants import MODEL_FEATURE_COLS  # noqa: E402
from src.models.analyst import load_analyst  # noqa: E402
from src.training.train_agent import create_trading_env, prepare_env_data  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreparedOOSData:
    df_5m: pd.DataFrame
    df_15m: pd.DataFrame
    df_45m: pd.DataFrame
    data_5m: np.ndarray
    data_15m: np.ndarray
    data_45m: np.ndarray
    close_prices: np.ndarray
    market_features: np.ndarray
    returns: Optional[np.ndarray]
    timestamps: np.ndarray
    ohlc: Optional[np.ndarray]
    rolling_lookback_data: Optional[np.ndarray]


def _resolve_path(cfg: Config, path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (cfg.paths.base_dir / path).resolve()
    return path


def _load_training_market_feat_stats(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Compute market feature mean/std from the TRAINING dataset (first 85% after windowing).

    This avoids normalizing OOS observations using OOS statistics (look-ahead).

    Returns:
        Tuple of (mean, std, warmup_data, train_market_features)
    """
    processed = cfg.paths.data_processed
    df_5m = pd.read_parquet(processed / "features_5m_normalized.parquet")
    df_15m = pd.read_parquet(processed / "features_15m_normalized.parquet")
    df_45m = pd.read_parquet(processed / "features_45m_normalized.parquet")

    required_feature_cols = list(MODEL_FEATURE_COLS)
    missing = [c for c in required_feature_cols if c not in df_5m.columns]
    if missing:
        raise ValueError(
            "Training processed data is missing required MODEL_FEATURE_COLS.\n"
            f"Missing: {sorted(missing)}\n"
            "Re-run `python scripts/core/run_pipeline.py` to regenerate `data/processed/*_normalized.parquet`."
        )

    lookback_5m = int(cfg.analyst.lookback_5m)
    lookback_15m = int(cfg.analyst.lookback_15m)
    lookback_45m = int(cfg.analyst.lookback_45m)

    _, _, _, close_prices, market_features, _, _ = prepare_env_data(
        df_5m,
        df_15m,
        df_45m,
        required_feature_cols,
        lookback_5m,
        lookback_15m,
        lookback_45m,
    )

    split_idx = int(0.85 * len(close_prices))
    train_market = market_features[:split_idx]
    market_feat_mean = train_market.mean(axis=0).astype(np.float32)
    market_feat_std = train_market.std(axis=0).astype(np.float32)
    market_feat_std = np.where(market_feat_std > 1e-8, market_feat_std, 1.0).astype(np.float32)
    rolling_window_size = cfg.normalization.rolling_window_size
    warmup = None
    if len(train_market) > 0:
        warmup = train_market[-rolling_window_size:].astype(np.float32)
    return market_feat_mean, market_feat_std, warmup, train_market.astype(np.float32)


def _apply_saved_normalizers(cfg: Config, df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_45m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalizer_dir = cfg.paths.models_analyst
    norm_5m_path = normalizer_dir / "normalizer_5m.pkl"
    norm_15m_path = normalizer_dir / "normalizer_15m.pkl"
    norm_45m_path = normalizer_dir / "normalizer_45m.pkl"

    for path in (norm_5m_path, norm_15m_path, norm_45m_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing saved normalizer: {path}\n"
                "Expected normalizers produced by `python scripts/core/run_pipeline.py` (Step 3b)."
            )

    n5 = FeatureNormalizer.load(norm_5m_path)
    n15 = FeatureNormalizer.load(norm_15m_path)
    n45 = FeatureNormalizer.load(norm_45m_path)

    # Transform only the columns each normalizer was fitted on; raw OHLC/ATR/session flags stay untouched.
    df_5m_n = n5.transform(df_5m)
    df_15m_n = n15.transform(df_15m)
    df_45m_n = n45.transform(df_45m)

    return df_5m_n, df_15m_n, df_45m_n


def _prepare_oos_from_csv(
    cfg: Config,
    csv_path: Path,
    rolling_warmup: Optional[np.ndarray] = None,
    training_baseline: Optional['TrainingBaseline'] = None,
) -> PreparedOOSData:
    logger.info("Loading OOS CSV: %s", csv_path)
    df_1m = load_ohlcv(csv_path, datetime_format=cfg.data.datetime_format)
    if len(df_1m) < 10_000:
        logger.warning("OOS CSV is quite small (%d rows). Backtest may be noisy.", len(df_1m))

    logger.info("Resampling 1m -> 5m/15m/45m")
    resampled = resample_all_timeframes(df_1m, cfg.data.timeframes)
    df_5m, df_15m, df_45m = resampled["5m"], resampled["15m"], resampled["45m"]

    logger.info("Engineering features (native TF first, then align)")
    feature_config = {
        "fractal_window": cfg.features.fractal_window,
        "sr_lookback": cfg.features.sr_lookback,
        "sma_period": cfg.features.sma_period,
        "ema_fast": cfg.features.ema_fast,
        "ema_slow": cfg.features.ema_slow,
        "chop_period": cfg.features.chop_period,
        "adx_period": cfg.features.adx_period,
        "atr_period": cfg.features.atr_period,
    }
    # v37: Pass training baseline for anchored OOD features
    df_5m = engineer_all_features(df_5m, feature_config, training_baseline=training_baseline)
    df_15m = engineer_all_features(df_15m, feature_config, training_baseline=training_baseline)
    df_45m = engineer_all_features(df_45m, feature_config, training_baseline=training_baseline)

    df_5m, df_15m, df_45m = align_timeframes(df_5m, df_15m, df_45m)

    # Ensure all three frames share the same non-NaN index.
    valid_5m = ~df_5m.isna().any(axis=1)
    valid_15m = ~df_15m.isna().any(axis=1)
    valid_45m = ~df_45m.isna().any(axis=1)
    common_valid = valid_5m & valid_15m & valid_45m
    df_5m = df_5m[common_valid]
    df_15m = df_15m[common_valid]
    df_45m = df_45m[common_valid]

    required_feature_cols = list(MODEL_FEATURE_COLS)
    for col in required_feature_cols:
        if col not in df_5m.columns:
            df_5m[col] = 0.0
        if col not in df_15m.columns:
            df_15m[col] = 0.0
        if col not in df_45m.columns:
            df_45m[col] = 0.0

    logger.info("Applying saved feature normalizers (from training run)")
    df_5m, df_15m, df_45m = _apply_saved_normalizers(cfg, df_5m, df_15m, df_45m)

    lookback_5m = int(cfg.analyst.lookback_5m)
    lookback_15m = int(cfg.analyst.lookback_15m)
    lookback_45m = int(cfg.analyst.lookback_45m)

    data_5m, data_15m, data_45m, close_prices, market_features, returns, rolling_lookback_data = prepare_env_data(
        df_5m,
        df_15m,
        df_45m,
        required_feature_cols,
        lookback_5m,
        lookback_15m,
        lookback_45m,
    )
    if rolling_warmup is not None and len(rolling_warmup) > 0:
        rolling_lookback_data = rolling_warmup.astype(np.float32)

    subsample_15m = 3
    subsample_45m = 9
    start_idx = max(
        lookback_5m,
        (lookback_15m - 1) * subsample_15m + 1,
        (lookback_45m - 1) * subsample_45m + 1,
    )
    n_samples = len(close_prices)

    if not isinstance(df_5m.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex after loading/resampling OOS data.")

    timestamps = (df_5m.index[start_idx:start_idx + n_samples].astype("int64") // 10**9).values

    ohlc: Optional[np.ndarray] = None
    if all(c in df_5m.columns for c in ("open", "high", "low", "close")):
        ohlc = df_5m[["open", "high", "low", "close"]].values[start_idx:start_idx + n_samples].astype(np.float32)

    logger.info(
        "OOS dataset ready | bars=%d | %s → %s",
        len(close_prices),
        pd.to_datetime(int(timestamps[0]), unit="s"),
        pd.to_datetime(int(timestamps[-1]), unit="s"),
    )

    return PreparedOOSData(
        df_5m=df_5m,
        df_15m=df_15m,
        df_45m=df_45m,
        data_5m=data_5m,
        data_15m=data_15m,
        data_45m=data_45m,
        close_prices=close_prices,
        market_features=market_features,
        returns=returns,
        timestamps=timestamps,
        ohlc=ohlc,
        rolling_lookback_data=rolling_lookback_data,
    )


def _log_distribution_shift(
    train_market_features: np.ndarray,
    oos_market_features: np.ndarray,
    feature_cols: list,
    oos_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    v36/v37 OOD FIX: Log distribution shift analysis comparing OOS to training.

    Returns dict with shift analysis for saving with results.
    """
    logger.info("=" * 60)
    logger.info("DISTRIBUTION SHIFT ANALYSIS (v36/v37 OOD Detection)")
    logger.info("=" * 60)

    # v37: Log training-anchored OOD score if available
    ood_score_stats = {}
    if oos_df is not None and 'ood_score' in oos_df.columns:
        ood_scores = oos_df['ood_score'].dropna()
        if len(ood_scores) > 0:
            ood_mean = float(ood_scores.mean())
            ood_std = float(ood_scores.std())
            ood_max = float(ood_scores.max())
            pct_high_ood = float((ood_scores > 0.5).mean() * 100)
            pct_critical_ood = float((ood_scores > 0.7).mean() * 100)

            logger.info("\nv37 Training-Anchored OOD Score:")
            logger.info("  Mean OOD Score:  %.3f (std: %.3f)", ood_mean, ood_std)
            logger.info("  Max OOD Score:   %.3f", ood_max)
            logger.info("  %% High OOD (>0.5):     %.1f%%", pct_high_ood)
            logger.info("  %% Critical OOD (>0.7): %.1f%%", pct_critical_ood)

            if ood_mean > 0.7:
                logger.warning("!! CRITICAL OOD: Mean score %.3f > 0.7", ood_mean)
                logger.warning("!! Model is likely to significantly underperform")
            elif ood_mean > 0.4:
                logger.warning("!! HIGH OOD: Mean score %.3f > 0.4", ood_mean)
                logger.warning("!! Model may underperform on this data")
            else:
                logger.info("  OOD score within acceptable range")

            ood_score_stats = {
                'ood_mean': ood_mean,
                'ood_std': ood_std,
                'ood_max': ood_max,
                'pct_high_ood': pct_high_ood,
                'pct_critical_ood': pct_critical_ood,
            }
    else:
        logger.info("\nv37 OOD Score: Not available (retrain to generate)")
        logger.info("  Using v36 rolling-window features only")

    # Compute basic volatility shift (first feature is typically ATR-related)
    train_vol_mean = float(train_market_features[:, 0].mean())
    train_vol_std = float(train_market_features[:, 0].std())
    oos_vol_mean = float(oos_market_features[:, 0].mean())

    vol_z_score = (oos_vol_mean - train_vol_mean) / (train_vol_std + 1e-8)

    logger.info("Volatility Shift:")
    logger.info("  Training mean: %.6f (std: %.6f)", train_vol_mean, train_vol_std)
    logger.info("  OOS mean:      %.6f", oos_vol_mean)
    logger.info("  Z-Score:       %.2f", vol_z_score)

    # Flag significant shifts
    if abs(vol_z_score) > 2.0:
        logger.warning("⚠️  SIGNIFICANT VOLATILITY SHIFT DETECTED (|z| > 2.0)")
        logger.warning("⚠️  Model may underperform due to out-of-distribution conditions")
    elif abs(vol_z_score) > 1.0:
        logger.warning("⚠️  Moderate volatility shift detected (|z| > 1.0)")
    else:
        logger.info("  ✓ Volatility within normal range")

    # Run full distribution shift analysis
    shift_analysis = {}
    try:
        # Limit features for analysis (avoid scipy import issues in some envs)
        n_features = min(len(feature_cols), train_market_features.shape[1], oos_market_features.shape[1])
        if n_features > 0:
            shift_analysis = analyze_distribution_shift(
                train_market_features[:, :n_features],
                oos_market_features[:, :n_features],
                feature_names=feature_cols[:n_features],
            )

            summary = shift_analysis.get('summary', {})
            n_significant = summary.get('n_significant_shifts', 0)
            pct_significant = summary.get('pct_significant', 0)

            logger.info("\nFeature-Level Shift Analysis:")
            logger.info("  Features analyzed:    %d", summary.get('n_features', 0))
            logger.info("  Significant shifts:   %d (%.1f%%)", n_significant, pct_significant)

            if pct_significant > 30:
                logger.warning("⚠️  HIGH DISTRIBUTION SHIFT: >30%% of features significantly different")
            elif pct_significant > 15:
                logger.warning("⚠️  Moderate distribution shift: >15%% of features different")

            # Log top shifted features
            top_features = summary.get('top_shifted_features', [])
            if top_features:
                logger.info("  Top shifted features: %s", ", ".join(top_features[:5]))

    except Exception as e:
        logger.warning("Could not run full distribution shift analysis: %s", e)

    logger.info("=" * 60)

    return {
        'volatility_z_score': vol_z_score,
        'train_vol_mean': train_vol_mean,
        'train_vol_std': train_vol_std,
        'oos_vol_mean': oos_vol_mean,
        'shift_analysis': shift_analysis,
        'ood_score_stats': ood_score_stats,  # v37 anchored OOD stats
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an OOS backtest on a custom CSV using an existing checkpoint.")
    parser.add_argument("--data-csv", type=str, required=True, help="Path to 1-minute OHLC CSV for OOS backtest.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SB3 PPO checkpoint .zip to evaluate.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum action confidence (0.0 disables).")
    parser.add_argument("--zero-costs", action="store_true", help="Run with spread=0 and slippage=0.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: results/<timestamp>_oos_csv).")
    return parser.parse_args()


def main() -> int:
    setup_logging(None, name=__name__)
    args = _parse_args()

    cfg = Config()
    cfg.device = get_device()

    data_csv = Path(args.data_csv).expanduser()
    checkpoint_path = _resolve_path(cfg, args.checkpoint)

    if not data_csv.exists():
        logger.error("Data CSV not found: %s", data_csv)
        return 2
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return 2

    if args.zero_costs:
        logger.info("ZERO COSTS MODE: spread=0, slippage=0")
        cfg.trading.spread_pips = 0.0
        cfg.trading.slippage_pips = 0.0

    # Disable noise for evaluation.
    cfg.trading.noise_level = 0.0

    # Compute training market feature stats for observation normalization + rolling warmup.
    market_feat_mean, market_feat_std, rolling_warmup, train_market_features = _load_training_market_feat_stats(cfg)
    if rolling_warmup is not None:
        logger.info("Rolling warmup loaded from training data: %d bars", len(rolling_warmup))

    # v37: Load training baseline for anchored OOD features
    training_baseline = None
    baseline_path = cfg.paths.models_agent / "training_baseline.json"
    if baseline_path.exists():
        try:
            training_baseline = TrainingBaseline.load(baseline_path)
            logger.info("v37 TrainingBaseline loaded from %s", baseline_path)
        except Exception as e:
            logger.warning("Failed to load training baseline: %s", e)
            logger.warning("Falling back to v36 rolling OOD features")
    else:
        logger.warning("No training baseline found at %s", baseline_path)
        logger.warning("v37 OOD features will use zero placeholders (retrain to generate baseline)")

    oos = _prepare_oos_from_csv(cfg, data_csv, rolling_warmup=rolling_warmup, training_baseline=training_baseline)

    # Load Analyst (if enabled).
    use_analyst = bool(getattr(cfg.trading, "use_analyst", True))
    analyst = None
    if use_analyst:
        analyst_path = cfg.paths.models_analyst / "best.pt"
        if not analyst_path.exists():
            raise FileNotFoundError(f"Analyst checkpoint not found: {analyst_path}")
        feature_dims = {"5m": len(MODEL_FEATURE_COLS), "15m": len(MODEL_FEATURE_COLS), "45m": len(MODEL_FEATURE_COLS)}
        analyst = load_analyst(str(analyst_path), feature_dims, device=cfg.device, freeze=True)
    else:
        logger.info("Analyst DISABLED (config.trading.use_analyst=False)")

    env = create_trading_env(
        oos.data_5m,
        oos.data_15m,
        oos.data_45m,
        oos.close_prices,
        oos.market_features,
        analyst_model=analyst,
        config=cfg.trading,
        device=cfg.device,
        market_feat_mean=market_feat_mean,
        market_feat_std=market_feat_std,
        returns=oos.returns,
        ohlc_data=oos.ohlc,
        timestamps=oos.timestamps,
        use_analyst=use_analyst,
        rolling_lookback_data=oos.rolling_lookback_data,
    )

    from stable_baselines3.common.monitor import Monitor

    monitor_env = Monitor(env)
    agent = SniperAgent.load(str(checkpoint_path), monitor_env, device="cpu")

    logger.info("Starting backtest...")
    start = time.time()
    results = run_backtest(
        agent=agent,
        env=monitor_env.unwrapped,
        initial_balance=float(cfg.trading.initial_balance),
        deterministic=True,
        min_action_confidence=float(args.min_confidence),
        # These are still passed for logging; Backtester uses env.spread_pips if set.
        spread_pips=float(getattr(cfg.trading, "spread_pips", 0.0)),
        sl_atr_multiplier=float(cfg.trading.sl_atr_multiplier),
        tp_atr_multiplier=float(cfg.trading.tp_atr_multiplier),
        use_stop_loss=bool(cfg.trading.use_stop_loss),
        use_take_profit=bool(cfg.trading.use_take_profit),
        min_hold_bars=int(cfg.trading.min_hold_bars),
        early_exit_profit_atr=float(cfg.trading.early_exit_profit_atr),
        break_even_atr=float(cfg.trading.break_even_atr),
    )
    elapsed = time.time() - start

    comparison = compare_with_baseline(results, oos.close_prices, initial_balance=float(cfg.trading.initial_balance))

    print_metrics_report(results.metrics, title="Agent Performance (OOS CSV)")
    print_comparison_report(comparison)
    logger.info("Backtest wall time: %.1fs", elapsed)

    # v36/v37 OOD FIX: Log distribution shift analysis
    shift_results = _log_distribution_shift(
        train_market_features=train_market_features,
        oos_market_features=oos.market_features,
        feature_cols=list(MODEL_FEATURE_COLS),
        oos_df=oos.df_5m,  # v37: Pass df for training-anchored OOD score logging
    )

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (cfg.paths.base_dir / "results" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_oos_csv")
    save_backtest_results(results, str(out_dir), comparison)

    # Save distribution shift analysis
    import json
    shift_path = out_dir / "distribution_shift.json"
    with open(shift_path, 'w') as f:
        # Convert numpy types for JSON serialization
        serializable = {
            'volatility_z_score': float(shift_results['volatility_z_score']),
            'train_vol_mean': float(shift_results['train_vol_mean']),
            'train_vol_std': float(shift_results['train_vol_std']),
            'oos_vol_mean': float(shift_results['oos_vol_mean']),
        }
        json.dump(serializable, f, indent=2)
    logger.info("Saved distribution shift analysis: %s", shift_path)
    logger.info("Saved backtest outputs: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
