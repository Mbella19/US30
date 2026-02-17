#!/usr/bin/env python3
"""
Prepare deployment artifacts needed for MT5 live inference.

Creates:
- `models/agent/market_feat_stats.npz`

This file stores the market-feature mean/std computed on the same TRAINING portion
used during agent training (first 85% of samples after windowing). The live bridge
uses these stats to normalize `market_features` exactly like TradingEnv does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def _load_parquet(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Expected DatetimeIndex in {path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MT5 bridge deployment artifacts")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    import numpy as np

    from config.settings import Config
    from src.live.bridge_constants import MARKET_FEATURE_COLS
    from src.utils.logging_config import setup_logging, get_logger

    logger = get_logger(__name__)

    if args.log_dir:
        setup_logging(args.log_dir, name=__name__)

    cfg = Config()

    # v28 FIX: Load ALL 3 timeframes to create multi-TF stats
    # This matches train_agent.py which concatenates 5m+15m+45m market features
    df_5m_path = cfg.paths.data_processed / "features_5m_normalized.parquet"
    df_15m_path = cfg.paths.data_processed / "features_15m_normalized.parquet"
    df_45m_path = cfg.paths.data_processed / "features_45m_normalized.parquet"

    df_5m = _load_parquet(df_5m_path)
    df_15m = _load_parquet(df_15m_path)
    df_45m = _load_parquet(df_45m_path)

    lookback_5m = int(cfg.analyst.lookback_5m)
    lookback_15m = int(cfg.analyst.lookback_15m)
    lookback_45m = int(cfg.analyst.lookback_45m)
    subsample_15m = 3
    subsample_45m = 9

    start_idx = max(
        lookback_5m,
        (lookback_15m - 1) * subsample_15m + 1,
        (lookback_45m - 1) * subsample_45m + 1,
    )
    n_samples = len(df_5m) - start_idx
    if n_samples <= 0:
        raise ValueError("Not enough rows in normalized data to build stats.")

    # Ensure all columns exist in all dataframes
    for df, tf_name in [(df_5m, "5m"), (df_15m, "15m"), (df_45m, "45m")]:
        for col in MARKET_FEATURE_COLS:
            if col not in df.columns:
                logger.warning("Missing market feature '%s' in %s (filling 0.0)", col, tf_name)
                df[col] = 0.0

    # v28 FIX: Extract market features from ALL 3 timeframes and concatenate
    # This matches train_agent.py which concatenates market features across 3 TFs
    mkt_5m = df_5m[MARKET_FEATURE_COLS].values[start_idx:start_idx + n_samples].astype(np.float32)
    mkt_15m = df_15m[MARKET_FEATURE_COLS].values[start_idx:start_idx + n_samples].astype(np.float32)
    mkt_45m = df_45m[MARKET_FEATURE_COLS].values[start_idx:start_idx + n_samples].astype(np.float32)

    # Concatenate: [5m_features, 15m_features, 45m_features]
    market_features = np.concatenate([mkt_5m, mkt_15m, mkt_45m], axis=1).astype(np.float32)
    logger.info(f"Multi-TF market features: {len(MARKET_FEATURE_COLS)} cols × 3 TFs = {market_features.shape[1]} total")

    split_idx = int(0.85 * len(market_features))
    train_market = market_features[:split_idx]

    mean = train_market.mean(axis=0).astype(np.float32)
    std = train_market.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)

    # Create column names for all 3 TFs
    cols_5m = [f"{c}_5m" for c in MARKET_FEATURE_COLS]
    cols_15m = [f"{c}_15m" for c in MARKET_FEATURE_COLS]
    cols_45m = [f"{c}_45m" for c in MARKET_FEATURE_COLS]
    all_cols = cols_5m + cols_15m + cols_45m

    out_path = cfg.paths.models_agent / "market_feat_stats.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, cols=np.array(all_cols), mean=mean, std=std)

    logger.info("Saved %s", out_path)
    logger.info(f"Total dimensions: {len(all_cols)} (market features × 3 timeframes)")
    logger.info(f"mean shape: {mean.shape}, std shape: {std.shape}")


if __name__ == "__main__":
    main()
