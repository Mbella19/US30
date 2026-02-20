"""
Training script for the PPO Sniper Agent.

Trains the RL agent using a frozen Market Analyst to provide
context vectors for decision making.

Memory-optimized for Apple M2 Silicon.

Features:
- Comprehensive logging of training progress
- Detailed reward and action statistics
- Training visualizations (reward curves, action distributions)
- Episode-level tracking and analysis
"""

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".mplconfig"),
)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List, Union
from datetime import datetime
import gc
import json

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from ..models.analyst import load_analyst, MarketAnalyst
from ..environments.trading_env import TradingEnv
from ..environments.env_factory import make_vec_env, prepare_env_kwargs_for_vectorization
from ..agents.sniper_agent import SniperAgent, create_agent
from ..utils.logging_config import setup_logging, get_logger
from ..utils.metrics import calculate_trading_metrics, TradingMetrics
from ..utils.callbacks import AgentTrainingLogger, GradientNormCallback
from ..data.features import (
    add_market_sessions,
    detect_fractals,
    detect_structure_breaks
)
from config.settings import config as default_config

logger = get_logger(__name__)


# =============================================================================
# Volatility Augmentation (v36 OOD Fix)
# =============================================================================

def augment_volatility_regimes(
    train_data: Tuple[np.ndarray, ...],
    vol_scales: Tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4),
    augment_ratio: float = 0.3,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Augment training data with different volatility scaling to improve OOD robustness.

    This addresses the distribution shift problem where the agent overfit to
    high-volatility training data and fails on lower-volatility test periods.

    Augmentation scales volatility-sensitive features (returns, volatility) by
    different factors to simulate varying market conditions during training.

    Args:
        train_data: Tuple of (data_5m, data_15m, data_45m, close_prices, market_features, returns)
        vol_scales: Volatility scaling factors to apply
        augment_ratio: Fraction of samples to augment (e.g., 0.3 = 30%)
        feature_cols: List of feature column names (to identify volatility columns)

    Returns:
        Augmented train_data tuple with same structure but scaled features
    """
    data_5m, data_15m, data_45m, close_prices, market_features, returns = train_data

    n_samples = len(close_prices)
    n_augment = int(n_samples * augment_ratio)

    if n_augment == 0:
        logger.info("Volatility augmentation skipped (augment_ratio=0)")
        return train_data

    logger.info(f"Applying volatility augmentation to {n_augment} samples ({augment_ratio*100:.0f}%)")
    logger.info(f"  Volatility scales: {vol_scales}")

    # Create copies to avoid modifying originals
    aug_data_5m = data_5m.copy()
    aug_data_15m = data_15m.copy()
    aug_data_45m = data_45m.copy()
    aug_market = market_features.copy()

    # Identify indices of volatility-related features
    # Default indices based on MODEL_FEATURE_COLS order:
    # 0: returns, 1: volatility (these are the main targets)
    returns_idx = 0
    volatility_idx = 1

    # In market features (3x timeframes concatenated):
    # Original order: atr, chop, adx, regime, sma_distance, ...
    # ATR is index 0 in each timeframe block
    from src.live.bridge_constants import MARKET_FEATURE_COLS
    n_market_cols = len(MARKET_FEATURE_COLS)

    # Find volatility-related indices in market features
    atr_indices = []
    vol_indices = []
    returns_market_indices = []
    for tf_offset in [0, n_market_cols, 2 * n_market_cols]:  # 5m, 15m, 45m
        if 'atr' in MARKET_FEATURE_COLS:
            atr_indices.append(tf_offset + MARKET_FEATURE_COLS.index('atr'))
        if 'volatility' in MARKET_FEATURE_COLS:
            vol_indices.append(tf_offset + MARKET_FEATURE_COLS.index('volatility'))
        if 'returns' in MARKET_FEATURE_COLS:
            returns_market_indices.append(tf_offset + MARKET_FEATURE_COLS.index('returns'))

    # Select random samples to augment
    np.random.seed(42)  # Reproducible augmentation
    aug_indices = np.random.choice(n_samples, n_augment, replace=False)

    # Count scaling applications
    scale_counts = {s: 0 for s in vol_scales}

    for idx in aug_indices:
        # Select a random scale (not 1.0 since that's no change)
        non_unity_scales = [s for s in vol_scales if s != 1.0]
        scale = np.random.choice(non_unity_scales)
        scale_counts[scale] = scale_counts.get(scale, 0) + 1

        # Scale model input features (returns, volatility)
        aug_data_5m[idx, :, returns_idx] *= scale
        aug_data_5m[idx, :, volatility_idx] *= scale
        aug_data_15m[idx, :, returns_idx] *= scale
        aug_data_15m[idx, :, volatility_idx] *= scale
        aug_data_45m[idx, :, returns_idx] *= scale
        aug_data_45m[idx, :, volatility_idx] *= scale

        # Scale market features
        for atr_idx in atr_indices:
            aug_market[idx, atr_idx] *= scale
        for vol_idx in vol_indices:
            aug_market[idx, vol_idx] *= scale
        for ret_idx in returns_market_indices:
            aug_market[idx, ret_idx] *= scale

    logger.info(f"  Augmentation complete. Scale distribution: {scale_counts}")

    return (aug_data_5m, aug_data_15m, aug_data_45m, close_prices, aug_market, returns)


def prepare_env_data(
    df_5m,
    df_15m,
    df_45m,
    feature_cols: list,
    lookback_5m: int = 48,
    lookback_15m: int = 16,
    lookback_45m: int = 6
) -> Tuple[
    np.ndarray,      # data_5m
    np.ndarray,      # data_15m
    np.ndarray,      # data_45m
    np.ndarray,      # close_prices
    np.ndarray,      # market_features
    np.ndarray,      # returns
    np.ndarray,      # rolling_lookback_data
]:
    """
    Prepare windowed data for the trading environment.
    
    FIXED: 15m and 45m data now correctly subsampled from the aligned 5m index.
    Since df_15m and df_45m are aligned to the 5m index via forward-fill,
    we subsample every 3rd bar for 15m and every 9th bar for 45m.

    Returns:
        Tuple of (data_5m, data_15m, data_45m, close_prices, market_features, returns, rolling_lookback_data)
    """
    # Subsampling ratios: how many 5m bars per higher TF bar
    subsample_15m = 3   # 3 x 5m = 15m
    subsample_45m = 9   # 9 x 5m = 45m
    
    # Calculate valid range accounting for subsampling.
    # Need enough bars for: 5m lookback, (15m lookback-1)*3+1, (45m lookback-1)*9+1.
    # This matches `src/training/precompute_analyst.py` and avoids dropping extra valid samples.
    start_idx = max(
        lookback_5m,
        (lookback_15m - 1) * subsample_15m + 1,
        (lookback_45m - 1) * subsample_45m + 1,
    )
    n_samples = len(df_5m) - start_idx

    logger.info(f"Preparing {n_samples} samples for environment")
    logger.info(f"  5m: {lookback_5m} bars = {lookback_5m * 5 / 60:.1f} hours")
    logger.info(f"  15m: {lookback_15m} bars = {lookback_15m * 15 / 60:.1f} hours (using {(lookback_15m - 1) * subsample_15m + 1} aligned indices)")
    logger.info(f"  45m: {lookback_45m} bars = {lookback_45m * 45 / 60:.1f} hours (using {(lookback_45m - 1) * subsample_45m + 1} aligned indices)")

    # Prepare windowed data
    data_5m = np.zeros((n_samples, lookback_5m, len(feature_cols)), dtype=np.float32)
    data_15m = np.zeros((n_samples, lookback_15m, len(feature_cols)), dtype=np.float32)
    data_45m = np.zeros((n_samples, lookback_45m, len(feature_cols)), dtype=np.float32)

    features_5m = df_5m[feature_cols].values.astype(np.float32)
    features_15m = df_15m[feature_cols].values.astype(np.float32)
    features_45m = df_45m[feature_cols].values.astype(np.float32)

    for i in range(n_samples):
        actual_idx = start_idx + i
        # 5m: direct indexing (include current candle)
        data_5m[i] = features_5m[actual_idx - lookback_5m + 1:actual_idx + 1]

        # FIXED: 15m - subsample every 3rd bar from aligned data, INCLUDING current candle
        # range() is exclusive at end, so we use actual_idx + 1 to include current
        idx_range_15m = list(range(
            actual_idx - (lookback_15m - 1) * subsample_15m,
            actual_idx + 1,
            subsample_15m
        ))
        data_15m[i] = features_15m[idx_range_15m]

        # FIXED: 45m - subsample every 9th bar from aligned data, INCLUDING current candle
        idx_range_45m = list(range(
            actual_idx - (lookback_45m - 1) * subsample_45m,
            actual_idx + 1,
            subsample_45m
        ))
        data_45m[i] = features_45m[idx_range_45m]

    # Close prices
    close_prices = df_5m['close'].values[start_idx:start_idx + n_samples].astype(np.float32)
    
    # Returns (for "Full Eyes" agent peripheral vision)
    # Using 'returns' column if available, else derive from prices
    if 'returns' in df_5m.columns:
        returns = df_5m['returns'].values[start_idx:start_idx + n_samples].astype(np.float32)
    else:
        # Calculate returns on the fly
        ret = df_5m['close'].pct_change().fillna(0).values
        returns = ret[start_idx:start_idx + n_samples].astype(np.float32)

    # Market features for observation (all 3 timeframes).
    # Keep in sync with MT5 bridge + backtests.
    from src.live.bridge_constants import MARKET_FEATURE_COLS

    market_cols = list(MARKET_FEATURE_COLS)
    available_cols = [c for c in market_cols if c in df_5m.columns]

    if len(available_cols) > 0:
        # Extract from ALL 3 timeframes and concatenate
        mkt_5m = df_5m[available_cols].values[start_idx:start_idx + n_samples].astype(np.float32)
        mkt_15m = df_15m[available_cols].values[start_idx:start_idx + n_samples].astype(np.float32)
        mkt_45m = df_45m[available_cols].values[start_idx:start_idx + n_samples].astype(np.float32)
        
        # Concatenate: [5m_features, 15m_features, 45m_features]
        market_features = np.concatenate([mkt_5m, mkt_15m, mkt_45m], axis=1).astype(np.float32)
        logger.info(f"Multi-timeframe market features: {len(available_cols)} cols × 3 TFs = {market_features.shape[1]} total")
        
        # Extract rolling lookback data (data BEFORE start_idx for warmup)
        rolling_window_size = default_config.normalization.rolling_window_size
        lookback_start = max(0, start_idx - rolling_window_size)
        lookback_5m = df_5m[available_cols].values[lookback_start:start_idx].astype(np.float32)
        lookback_15m = df_15m[available_cols].values[lookback_start:start_idx].astype(np.float32)
        lookback_45m = df_45m[available_cols].values[lookback_start:start_idx].astype(np.float32)
        rolling_lookback_data = np.concatenate([lookback_5m, lookback_15m, lookback_45m], axis=1).astype(np.float32)
        logger.info(f"Rolling lookback data: {len(rolling_lookback_data)} bars before start_idx")
    else:
        # Create dummy features if not available (len(MARKET_FEATURE_COLS) × 3)
        market_features = np.zeros((n_samples, len(MARKET_FEATURE_COLS) * 3), dtype=np.float32)
        market_features[:, 0] = 0.001  # Default ATR (5m)
        market_features[:, 1] = 50.0   # Default CHOP
        market_features[:, 2] = 20.0   # Default ADX
        rolling_lookback_data = None

    return data_5m, data_15m, data_45m, close_prices, market_features, returns, rolling_lookback_data


def create_trading_env(
    data_5m: np.ndarray,
    data_15m: np.ndarray,
    data_45m: np.ndarray,
    close_prices: np.ndarray,
    market_features: np.ndarray,
    analyst_model: Optional[MarketAnalyst] = None,
    config: Optional[object] = None,
    device: Optional[torch.device] = None,
    market_feat_mean: Optional[np.ndarray] = None,
    market_feat_std: Optional[np.ndarray] = None,
    precomputed_analyst_cache: Optional[dict] = None,
    ohlc_data: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    returns: Optional[np.ndarray] = None,
    use_analyst: bool = True,
    rolling_lookback_data: Optional[np.ndarray] = None,
) -> TradingEnv:
    """
    Create the trading environment.

    Args:
        data_5m: 5-minute window data (base timeframe)
        data_15m: 15-minute window data (medium timeframe)
        data_45m: 45-minute window data (trend timeframe)
        close_prices: Close prices for PnL
        market_features: Features for reward shaping
        analyst_model: Frozen Market Analyst
        config: TradingConfig
        device: Torch device

    Returns:
        TradingEnv instance
    """
    # Default configuration (matches config/settings.py TradingConfig)
    spread_pips = 10.0              # config.trading.spread_pips
    slippage_pips = 0.0             # config.trading.slippage_pips
    fomo_threshold_atr = 4.0        # config.trading.fomo_threshold_atr
    fomo_lookback_bars = 24         # config.trading.fomo_lookback_bars
    chop_threshold = 80.0           # config.trading.chop_threshold
    max_steps = 500
    reward_scaling = 0.01           # config.trading.reward_scaling
    context_dim = 32                # config.analyst.context_dim
    trade_entry_bonus = 0.1         # config.trading.trade_entry_bonus
    noise_level = 0.05              # Moderate regularization noise
    # Alpha-Based Reward and Symmetric PnL Scaling
    profit_scaling = 0.01           # config.trading.profit_scaling
    loss_scaling = 0.01             # config.trading.loss_scaling
    use_alpha_reward = True         # config.trading.use_alpha_reward
    alpha_baseline_exposure = 0.7   # config.trading.alpha_baseline_exposure

    # Risk Management defaults
    sl_atr_multiplier = 2.0         # config.trading.sl_atr_multiplier
    tp_atr_multiplier = 6.0         # config.trading.tp_atr_multiplier
    use_stop_loss = True            # config.trading.use_stop_loss
    use_take_profit = True          # config.trading.use_take_profit

    # Volatility Sizing (Dollar-based risk)
    volatility_sizing = True
    risk_per_trade = 100.0          # config.trading.risk_per_trade

    # Minimum Hold Time
    min_hold_bars = 0               # config.trading.min_hold_bars
    # Profit-based early exit override
    early_exit_profit_atr = 0.0     # config.trading.early_exit_profit_atr
    # Break-even stop loss
    break_even_atr = 2.0            # config.trading.break_even_atr
    # Opportunity Cost
    opportunity_cost_multiplier = 0.0  # config.trading.opportunity_cost_multiplier
    opportunity_cost_cap = 0.2      # config.trading.opportunity_cost_cap
    rolling_norm_min_samples = 1

    if config is not None:
        # FIX: Access config.trading for trading parameters (not config directly)
        trading_cfg = getattr(config, 'trading', config)
        spread_pips = getattr(trading_cfg, 'spread_pips', spread_pips)
        slippage_pips = getattr(trading_cfg, 'slippage_pips', slippage_pips)
        fomo_threshold_atr = getattr(trading_cfg, 'fomo_threshold_atr', fomo_threshold_atr)
        fomo_lookback_bars = getattr(trading_cfg, 'fomo_lookback_bars', fomo_lookback_bars)
        chop_threshold = getattr(trading_cfg, 'chop_threshold', chop_threshold)
        max_steps = getattr(trading_cfg, 'max_steps_per_episode', max_steps)
        reward_scaling = getattr(trading_cfg, 'reward_scaling', reward_scaling)
        # Risk Management
        sl_atr_multiplier = getattr(trading_cfg, 'sl_atr_multiplier', sl_atr_multiplier)
        tp_atr_multiplier = getattr(trading_cfg, 'tp_atr_multiplier', tp_atr_multiplier)
        use_stop_loss = getattr(trading_cfg, 'use_stop_loss', use_stop_loss)
        use_take_profit = getattr(trading_cfg, 'use_take_profit', use_take_profit)
        # Trade entry bonus
        trade_entry_bonus = getattr(trading_cfg, 'trade_entry_bonus', trade_entry_bonus)
        noise_level = getattr(trading_cfg, 'noise_level', noise_level)
        # Alpha-Based Reward and Asymmetric PnL Scaling
        profit_scaling = getattr(trading_cfg, 'profit_scaling', profit_scaling)
        loss_scaling = getattr(trading_cfg, 'loss_scaling', loss_scaling)
        use_alpha_reward = getattr(trading_cfg, 'use_alpha_reward', use_alpha_reward)
        alpha_baseline_exposure = getattr(trading_cfg, 'alpha_baseline_exposure', alpha_baseline_exposure)
        # Minimum Hold Time
        min_hold_bars = getattr(trading_cfg, 'min_hold_bars', min_hold_bars)
        # Profit-based early exit override
        early_exit_profit_atr = getattr(trading_cfg, 'early_exit_profit_atr', early_exit_profit_atr)
        # Break-even stop loss
        break_even_atr = getattr(trading_cfg, 'break_even_atr', break_even_atr)
        # Opportunity Cost
        opportunity_cost_multiplier = getattr(trading_cfg, 'opportunity_cost_multiplier', opportunity_cost_multiplier)
        opportunity_cost_cap = getattr(trading_cfg, 'opportunity_cost_cap', opportunity_cost_cap)
        rolling_norm_min_samples = getattr(trading_cfg, 'rolling_norm_min_samples', rolling_norm_min_samples)

        # Log config values to verify they're applied
        logger.info(f"Config applied: reward_scaling={reward_scaling}, "
                    f"slippage_pips={slippage_pips}, trade_entry_bonus={trade_entry_bonus}, "
                    f"noise_level={noise_level}, min_hold_bars={min_hold_bars}")
        # Log SL/TP values to verify 1:2 R/R is applied
        logger.info(f"SL/TP config: sl_atr_multiplier={sl_atr_multiplier}, tp_atr_multiplier={tp_atr_multiplier} "
                    f"(R/R ratio: 1:{tp_atr_multiplier/sl_atr_multiplier:.1f})")

    if analyst_model is not None:
        context_dim = analyst_model.context_dim
        # Get num_classes from analyst (binary=2, multi-class=3)
        num_classes = getattr(analyst_model, 'num_classes', 2)
    else:
        num_classes = 2  # Default to binary

    env = TradingEnv(
        data_5m=data_5m,
        data_15m=data_15m,
        data_45m=data_45m,
        close_prices=close_prices,
        market_features=market_features,
        analyst_model=analyst_model,
        context_dim=context_dim,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        fomo_threshold_atr=fomo_threshold_atr,
        fomo_lookback_bars=fomo_lookback_bars,
        chop_threshold=chop_threshold,
        max_steps=max_steps,
        reward_scaling=reward_scaling,
        trade_entry_bonus=trade_entry_bonus,
        # Alpha-Based Reward and Asymmetric PnL Scaling
        profit_scaling=profit_scaling,
        loss_scaling=loss_scaling,
        use_alpha_reward=use_alpha_reward,
        alpha_baseline_exposure=alpha_baseline_exposure,
        device=device,
        noise_level=noise_level,
        market_feat_mean=market_feat_mean,
        market_feat_std=market_feat_std,
        # Risk Management
        sl_atr_multiplier=sl_atr_multiplier,
        tp_atr_multiplier=tp_atr_multiplier,
        use_stop_loss=use_stop_loss,
        use_take_profit=use_take_profit,
        # Volatility Sizing (Dollar-based risk)
        volatility_sizing=volatility_sizing,
        risk_per_trade=risk_per_trade,
        # Classification mode
        num_classes=num_classes,
        # Minimum Hold Time
        min_hold_bars=min_hold_bars,
        # Profit-based early exit override
        early_exit_profit_atr=early_exit_profit_atr,
        # Break-even stop loss
        break_even_atr=break_even_atr,
        # Opportunity Cost
        opportunity_cost_multiplier=opportunity_cost_multiplier,
        opportunity_cost_cap=opportunity_cost_cap,
        # Pre-computed Analyst cache
        precomputed_analyst_cache=precomputed_analyst_cache,
        # Visualization data
        ohlc_data=ohlc_data,
        timestamps=timestamps,
        # Full Eyes Features
        returns=returns,
        agent_lookback_window=getattr(trading_cfg, 'agent_lookback_window', 6) if config is not None else 6,
        # Toggle Analyst usage
        use_analyst=use_analyst,
        # Rolling window warmup data
        rolling_lookback_data=rolling_lookback_data,
        rolling_norm_min_samples=rolling_norm_min_samples,
    )

    return env


def train_agent(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_45m: pd.DataFrame,
    feature_cols: list,
    analyst_path: str,
    save_path: str,
    config: Optional[object] = None,
    device: Optional[torch.device] = None,
    total_timesteps: int = 500_000,
    resume_path: Optional[str] = None
) -> Tuple[SniperAgent, Dict]:
    """
    Main function to train the PPO Sniper Agent.

    Args:
        df_5m: 5-minute DataFrame with features (base timeframe)
        df_15m: 15-minute DataFrame with features (medium timeframe)
        df_45m: 45-minute DataFrame with features (trend timeframe)
        feature_cols: Feature columns used
        analyst_path: Path to trained analyst model
        save_path: Path to save agent
        config: Configuration object
        device: Torch device
        total_timesteps: Total training timesteps
        resume_path: Optional path to resume from checkpoint

    Returns:
        Tuple of (trained agent, training info)
    """
    # Setup logging for this run
    log_dir = Path(save_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(log_dir), name=__name__)

    # PPO/SB3 training is forced to CPU for stability and to avoid GPU/MPS usage.
    if device is not None and getattr(device, "type", None) != "cpu":
        logger.warning(f"Overriding requested device {device} -> cpu for PPO training.")
    device = torch.device("cpu")
    logger.info("Training PPO agent on device: cpu")

    # Check if analyst should be used (toggle from config)
    use_analyst = getattr(config.trading, 'use_analyst', True) if config else True

    from src.live.bridge_constants import MODEL_FEATURE_COLS
    required_feature_cols = list(MODEL_FEATURE_COLS)

    if feature_cols != required_feature_cols:
        logger.warning("Overriding `feature_cols` with canonical MODEL_FEATURE_COLS ordering.")
        feature_cols = required_feature_cols

    if use_analyst:
        # Load frozen analyst
        logger.info(f"Loading analyst from {analyst_path}")

        # TCNAnalyst expects true timeframe keys in this system
        feature_dims = {
            '5m': len(feature_cols),
            '15m': len(feature_cols),
            '45m': len(feature_cols)
        }
        analyst = load_analyst(analyst_path, feature_dims, device, freeze=True)
        logger.info("Analyst loaded and frozen")

        # Log analyst info
        logger.info(f"Analyst context_dim: {analyst.context_dim}")
        logger.info(f"Analyst parameters: {sum(p.numel() for p in analyst.parameters()):,} (frozen)")
    else:
        logger.info("=" * 70)
        logger.info("ANALYST DISABLED (use_analyst=False)")
        logger.info("Agent will train with market features only (no analyst context)")
        logger.info("=" * 70)
        analyst = None

    # Prepare data
    logger.info("Preparing environment data...")
    
    # Add Market Sessions
    logger.info("Adding market session features...")
    df_5m = add_market_sessions(df_5m)
    df_15m = add_market_sessions(df_15m)
    df_45m = add_market_sessions(df_45m)

    # Add Structure Features (BOS/CHoCH)
    logger.info("Adding structure features (BOS/CHoCH)...")
    for df in [df_5m, df_15m, df_45m]:
        f_high, f_low = detect_fractals(df)
        struct_df = detect_structure_breaks(df, f_high, f_low)
        for col in struct_df.columns:
            df[col] = struct_df[col]

    missing_cols = [
        c for c in required_feature_cols
        if c not in df_5m.columns or c not in df_15m.columns or c not in df_45m.columns
    ]
    if missing_cols:
        raise ValueError(
            "Missing required model feature columns after feature augmentation. "
            "Re-run the pipeline to regenerate features.\n"
            f"Missing: {sorted(missing_cols)}"
        )

    # NOTE: Do NOT append session/structure columns to `feature_cols` here.
    # `feature_cols` must match what the Analyst checkpoint was trained on.
    # Session/structure columns can still be used by the agent via `market_features`
    # (see `prepare_env_data()`).

    lookback_5m = config.analyst.lookback_5m
    lookback_15m = config.analyst.lookback_15m
    lookback_45m = config.analyst.lookback_45m

    data_5m, data_15m, data_45m, close_prices, market_features, returns, rolling_lookback_data = prepare_env_data(
        df_5m, df_15m, df_45m, feature_cols,
        lookback_5m, lookback_15m, lookback_45m,
    )

    logger.info(f"Data shapes: 5m={data_5m.shape}, 15m={data_15m.shape}, 45m={data_45m.shape}")
    logger.info(f"Price range: {close_prices.min():.5f} - {close_prices.max():.5f}")

    # Extract real OHLC data for visualization
    # Calculate start_idx to match prepare_env_data (5m base with 15m/45m subsampling)
    subsample_15m = 3
    subsample_45m = 9
    start_idx = max(
        lookback_5m,
        (lookback_15m - 1) * subsample_15m + 1,
        (lookback_45m - 1) * subsample_45m + 1,
    )
    n_samples = len(close_prices)
    
    ohlc_data = None
    timestamps = None
    if all(col in df_5m.columns for col in ['open', 'high', 'low', 'close']):
        ohlc_data = df_5m[['open', 'high', 'low', 'close']].values[start_idx:start_idx + n_samples].astype(np.float32)
        logger.info(f"OHLC data extracted: {ohlc_data.shape}, range: {ohlc_data[:, 3].min():.5f} - {ohlc_data[:, 3].max():.5f}")

    if df_5m.index.dtype == 'datetime64[ns]' or hasattr(df_5m.index, 'to_pydatetime'):
        try:
            timestamps = (df_5m.index[start_idx:start_idx + n_samples].astype('int64') // 10**9).values
            logger.info(f"Timestamps extracted: {len(timestamps)}")
        except Exception as e:
            logger.warning(f"Failed to extract timestamps: {e}")

    # Split into train / OOS using date-based splits from DataSplitConfig.
    # v38 FIX: Use config dates instead of hardcoded percentages to match
    # the normalizer fitting (step_3b) and backtest (step_6) boundaries.
    # Everything after train_end_date is out-of-sample.
    n_total = len(close_prices)

    if config and hasattr(config, 'data_split') and config.data_split.use_date_splits:
        train_end_date = pd.Timestamp(config.data_split.train_end_date)
        # Map date to index in the trimmed array.
        # close_prices[i] corresponds to df_5m.iloc[start_idx + i].
        trimmed_index = df_5m.index[start_idx:start_idx + n_total]
        train_mask = trimmed_index < train_end_date
        train_split = int(train_mask.sum())

        # Validation split (if configured, otherwise all OOS is "test")
        if hasattr(config.data_split, 'validation_end_date') and config.data_split.validation_end_date:
            val_end_date = pd.Timestamp(config.data_split.validation_end_date)
            val_mask = trimmed_index < val_end_date
            val_split = int(val_mask.sum())
        else:
            val_split = train_split  # No separate validation — all OOS is test

        logger.info(f"Date-based split (train_end={config.data_split.train_end_date}):")
    else:
        # Fallback to percentage-based splits
        train_split = int(0.70 * n_total)
        val_split = int(0.85 * n_total)
        logger.info(f"Percentage-based split (70/15/15):")

    logger.info(f"  Training:   0 to {train_split} ({train_split} samples, {train_split/n_total*100:.1f}%)")
    logger.info(f"  Validation: {train_split} to {val_split} ({val_split - train_split} samples, {(val_split-train_split)/n_total*100:.1f}%)")
    logger.info(f"  Test:       {val_split} to {n_total} ({n_total - val_split} samples, {(n_total-val_split)/n_total*100:.1f}%)")

    # Use train_split for backward compatibility with existing code
    split_idx = train_split

    # v38 FIX: Compute market feature normalization stats from TRAINING data
    # BEFORE volatility augmentation, so stats reflect the true distribution.
    train_market_features_raw = market_features[:split_idx]
    market_feat_mean = train_market_features_raw.mean(axis=0).astype(np.float32)
    market_feat_std = train_market_features_raw.std(axis=0).astype(np.float32)
    market_feat_std = np.where(market_feat_std > 1e-8, market_feat_std, 1.0).astype(np.float32)

    logger.info("Market feature normalization stats (from raw training data, pre-augmentation):")
    logger.info(f"  Mean: {market_feat_mean}")
    logger.info(f"  Std:  {market_feat_std}")

    train_data = (
        data_5m[:split_idx],
        data_15m[:split_idx],
        data_45m[:split_idx],
        close_prices[:split_idx],
        market_features[:split_idx],
        returns[:split_idx] if returns is not None else None,
    )

    # v36 OOD FIX: Apply volatility augmentation to training data
    # This exposes the agent to different volatility regimes during training,
    # improving robustness to distribution shifts in live/OOS data
    use_vol_augmentation = getattr(config.trading, 'use_volatility_augmentation', True) if config else True
    vol_augment_ratio = getattr(config.trading, 'volatility_augment_ratio', 0.3) if config else 0.3

    if use_vol_augmentation and vol_augment_ratio > 0:
        train_data = augment_volatility_regimes(
            train_data,
            vol_scales=(0.6, 0.8, 1.0, 1.2, 1.4),
            augment_ratio=vol_augment_ratio,
            feature_cols=feature_cols,
        )
        logger.info("Volatility augmentation applied to training data (v36 OOD fix)")
    else:
        logger.info("Volatility augmentation disabled")

    # Validation data (for post-training evaluation)
    val_data = (
        data_5m[train_split:val_split],
        data_15m[train_split:val_split],
        data_45m[train_split:val_split],
        close_prices[train_split:val_split],
        market_features[train_split:val_split],
        returns[train_split:val_split] if returns is not None else None,
    )

    # Test data (completely held out - for final evaluation only)
    test_data = (
        data_5m[val_split:],
        data_15m[val_split:],
        data_45m[val_split:],
        close_prices[val_split:],
        market_features[val_split:],
        returns[val_split:] if returns is not None else None,
    )

    # For backward compatibility, eval_data points to validation set
    eval_data = val_data

    # Split OHLC data for train/val/test
    train_ohlc = ohlc_data[:split_idx] if ohlc_data is not None else None
    val_ohlc = ohlc_data[train_split:val_split] if ohlc_data is not None else None
    test_ohlc = ohlc_data[val_split:] if ohlc_data is not None else None
    eval_ohlc = val_ohlc  # Backward compatibility
    train_timestamps = timestamps[:split_idx] if timestamps is not None else None
    val_timestamps = timestamps[train_split:val_split] if timestamps is not None else None
    test_timestamps = timestamps[val_split:] if timestamps is not None else None
    eval_timestamps = val_timestamps  # Backward compatibility

    # Live-style rolling normalization warmup:
    # - Train env uses pre-start lookback data from prepare_env_data (if available).
    # - Eval env uses the last rolling window from the training segment.
    rolling_window_size = config.normalization.rolling_window_size
    rolling_warmup_train = rolling_lookback_data
    rolling_warmup_eval = None
    if split_idx > 0 and market_features is not None:
        warmup_start = max(0, split_idx - rolling_window_size)
        rolling_warmup_eval = market_features[warmup_start:split_idx].astype(np.float32)

    # Try to load pre-computed Analyst cache for sequential context (only if use_analyst=True)
    analyst_cache_path = Path(save_path).parent.parent / 'data' / 'processed' / 'analyst_cache.npz'
    train_analyst_cache = None
    eval_analyst_cache = None

    if use_analyst:
        if analyst_cache_path.exists():
            logger.info(f"Loading pre-computed Analyst cache from {analyst_cache_path}")
            try:
                from .precompute_analyst import load_cached_analyst_outputs
                full_cache = load_cached_analyst_outputs(str(analyst_cache_path))

                # Validate cache length matches expected samples (defense against lookback mismatch)
                expected_samples = len(close_prices)
                cache_samples = len(full_cache['contexts'])
                if cache_samples != expected_samples:
                    logger.warning(
                        f"Analyst cache mismatch: cache has {cache_samples} samples vs expected {expected_samples}. "
                        f"Cache lookbacks=({full_cache['lookback_5m']},{full_cache['lookback_15m']},{full_cache['lookback_45m']}), "
                        f"Config lookbacks=({lookback_5m},{lookback_15m},{lookback_45m}). "
                        f"Regenerate cache with: python src/training/precompute_analyst.py"
                    )
                    logger.warning("Ignoring misaligned cache - falling back to on-the-fly Analyst inference")
                    raise ValueError("Cache mismatch - skip to fallback")

                # Split cache to match train/eval split
                cache_split_idx = split_idx
                train_analyst_cache = {
                    'contexts': full_cache['contexts'][:cache_split_idx],
                    'probs': full_cache['probs'][:cache_split_idx],
                    # DISABLED ACTIVATIONS TO SAVE MEMORY (OOM Protection)
                    'activations_15m': None,
                    'activations_1h': None,
                    'activations_4h': None,
                }
                eval_analyst_cache = {
                    'contexts': full_cache['contexts'][cache_split_idx:],
                    'probs': full_cache['probs'][cache_split_idx:],
                    # DISABLED ACTIVATIONS TO SAVE MEMORY (OOM Protection)
                    'activations_15m': None,
                    'activations_1h': None,
                    'activations_4h': None,
                }
                logger.info(f"Using sequential Analyst context: train={len(train_analyst_cache['contexts'])}, eval={len(eval_analyst_cache['contexts'])}")
            except Exception as e:
                logger.warning(f"Failed to load Analyst cache: {e}")
                logger.info("Falling back to standard precomputation")
        else:
            logger.info("No pre-computed Analyst cache found. Using standard precomputation.")
            logger.info(f"To enable sequential context, run: python src/training/precompute_analyst.py")
    else:
        logger.info("Skipping Analyst cache loading (use_analyst=False)")

    # Get vectorization settings from config
    agent_cfg = getattr(config, 'agent', None) if config else None
    n_envs = getattr(agent_cfg, 'n_envs', 1) if agent_cfg else 1
    use_subproc = getattr(agent_cfg, 'use_subproc', True) if agent_cfg else True

    # Create training environment
    viz_timestamps = train_timestamps

    if n_envs > 1:
        # Vectorized training for faster sample collection
        logger.info(f"Creating {n_envs} vectorized training environments...")

        # Prepare picklable env kwargs
        train_env_kwargs = prepare_env_kwargs_for_vectorization(
            data_5m=train_data[0],
            data_15m=train_data[1],
            data_45m=train_data[2],
            close_prices=train_data[3],
            market_features=train_data[4],
            returns=train_data[5],
            config=config,
            market_feat_mean=market_feat_mean,
            market_feat_std=market_feat_std,
            precomputed_analyst_cache=train_analyst_cache,
            ohlc_data=train_ohlc,
            timestamps=viz_timestamps,
            use_analyst=use_analyst,
            rolling_lookback_data=rolling_warmup_train,
        )

        # Create vectorized environment
        train_env = make_vec_env(
            n_envs=n_envs,
            use_subproc=use_subproc,
            env_kwargs=train_env_kwargs,
            seed=config.seed if config else 42,
        )

        logger.info(f"Vectorized training environment created: {type(train_env).__name__} with {n_envs} envs")
    else:
        # Single environment (original behavior)
        train_env = create_trading_env(
            data_5m=train_data[0],
            data_15m=train_data[1],
            data_45m=train_data[2],
            close_prices=train_data[3],
            market_features=train_data[4],
            returns=train_data[5],
            analyst_model=analyst,
            config=config,
            device=device,
            market_feat_mean=market_feat_mean,
            market_feat_std=market_feat_std,
            precomputed_analyst_cache=train_analyst_cache,
            ohlc_data=train_ohlc,
            timestamps=viz_timestamps,
            use_analyst=use_analyst,
            rolling_lookback_data=rolling_warmup_train,
        )
        # Wrap single env in Monitor
        train_env = Monitor(train_env)

    # Eval environment is always single-env for consistent evaluation
    logger.info("Creating evaluation environment (single env for consistent metrics)...")
    eval_env = create_trading_env(
        data_5m=eval_data[0],
        data_15m=eval_data[1],
        data_45m=eval_data[2],
        close_prices=eval_data[3],
        market_features=eval_data[4],
        returns=eval_data[5],
        analyst_model=analyst,
        config=config,
        device=device,
        market_feat_mean=market_feat_mean,
        market_feat_std=market_feat_std,
        precomputed_analyst_cache=eval_analyst_cache,
        ohlc_data=eval_ohlc,            # Real OHLC for visualization
        timestamps=eval_timestamps,      # Real timestamps for visualization
        use_analyst=use_analyst,
        rolling_lookback_data=rolling_warmup_eval,
    )
    eval_env = Monitor(eval_env)

    # Log environment info
    if n_envs > 1:
        # For vectorized env, get info from a single env
        logger.info(f"Observation space: {train_env.observation_space}")
        logger.info(f"Action space: {train_env.action_space}")
        logger.info(f"Training with {n_envs} parallel environments (effective batch: {n_envs * 2048} steps/update)")
    else:
        logger.info(f"Observation space: {train_env.observation_space}")
        logger.info(f"Action space: {train_env.action_space}")

    # Create agent
    reset_timesteps = True
    remaining_timesteps = total_timesteps

    if resume_path:
        resume_p = Path(str(resume_path)).expanduser()
        if not resume_p.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found (or not a file): {resume_p}\n"
                "Expected a SB3 .zip checkpoint like: models/checkpoints/sniper_model_7400000_steps.zip"
            )

        logger.info(f"Resuming agent from checkpoint: {resume_p}")
        try:
            agent = SniperAgent.load(str(resume_p), env=train_env, device=device)
        except Exception as e:
            logger.error(
                "Failed to resume from checkpoint. Most common causes:\n"
                "- Observation space changed (e.g. added/removed features, changed `agent_lookback_window`, "
                "or market feature columns)\n"
                "- Action space changed\n"
                f"Checkpoint: {resume_p}\n"
                f"Env obs shape: {getattr(train_env.observation_space, 'shape', None)}\n"
                f"Env action space: {train_env.action_space}\n"
                f"Error: {e}"
            )
            raise

        # Force-update Exploration Rate (Entropy Coefficient) from current config
        # This allows "shock therapy" (increasing exploration) on resumed models
        if config is not None and hasattr(config, "agent") and hasattr(config.agent, 'ent_coef'):
            current_ent_coef = getattr(agent.model, 'ent_coef', None)
            new_ent_coef = config.agent.ent_coef
            agent.model.ent_coef = new_ent_coef
            logger.info(f"Exploration Rate (Entropy) UPDATED: {current_ent_coef} -> {new_ent_coef}")

        # Force-update gamma from current config when resuming.
        # SB3 checkpoints load hyperparameters from the saved model; this ensures
        # the resumed run actually uses the new discount factor.
        if config is not None and hasattr(config, "agent") and hasattr(config.agent, "gamma"):
            current_gamma = getattr(agent.model, "gamma", None)
            new_gamma = config.agent.gamma
            agent.model.gamma = new_gamma
            logger.info(f"Gamma UPDATED: {current_gamma} -> {new_gamma}")

        # Calculate remaining steps
        current_timesteps = agent.model.num_timesteps
        # Ensure we run for at least 10k steps if completed or close to completion,
        # otherwise run until total_timesteps is reached.
        # If user wants to EXTEND training, they should increase total_timesteps in config.
        remaining_timesteps = max(10000, total_timesteps - current_timesteps)
        reset_timesteps = False

        logger.info(f"Resumed from step {current_timesteps:,}. Target: {total_timesteps:,}.")
        logger.info(f"Remaining timesteps: {remaining_timesteps:,}")
        logger.info("Learning Rate Schedule will CONTINUE from current step (NOT reset).")

        # Enable TensorBoard on resumed model
        agent.model.tensorboard_log = str(log_dir / "tb_logs")

    else:
        logger.info("Creating PPO agent...")
        tb_log_dir = str(log_dir / "tb_logs")
        agent = create_agent(train_env, config.agent if config else None, device=device, tensorboard_log=tb_log_dir)
        remaining_timesteps = total_timesteps
        reset_timesteps = True

    # Create training logger callback
    training_callback = AgentTrainingLogger(
        log_dir=str(log_dir),
        log_freq=5000,
        verbose=1
    )

    # Create gradient norm callback (Level 2: per-layer gradient monitoring)
    gradient_callback = GradientNormCallback(
        log_dir=str(log_dir),
        log_freq=50_000,
        verbose=1,
    )

    from stable_baselines3.common.callbacks import CallbackList
    all_callbacks = CallbackList([training_callback, gradient_callback])

    # Train
    logger.info(f"Starting training for {remaining_timesteps:,} timesteps...")
    logger.info(f"TensorBoard: tensorboard --logdir {log_dir / 'tb_logs'}")
    logger.info("-" * 70)

    # v35 FIX: Remove eval_env from training loop to prevent implicit look-ahead
    # Evaluation now happens ONLY after training completes (line 845)
    # This prevents the value function from learning OOS-specific patterns
    training_info = agent.train(
        total_timesteps=remaining_timesteps,
        eval_env=None,  # No eval during training - prevents look-ahead bias
        eval_freq=0,    # Disabled - eval only happens post-training
        save_path=save_path,
        callback=all_callbacks,
        reset_num_timesteps=reset_timesteps
    )

    # Get metrics from callback
    training_info['callback_metrics'] = training_callback.get_metrics()

    # Final evaluation
    logger.info("=" * 70)
    logger.info("Running final evaluation...")
    eval_results = agent.evaluate(eval_env, n_episodes=20)
    training_info['final_eval'] = eval_results

    logger.info("-" * 70)
    logger.info("FINAL EVALUATION RESULTS:")
    logger.info(f"  Mean Reward: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")
    logger.info(f"  Mean PnL: {eval_results['mean_pnl']:.2f} pips")
    logger.info(f"  Win Rate: {eval_results.get('win_rate', 0)*100:.1f}%")
    logger.info(f"  Mean Trades per Episode: {eval_results.get('mean_trades', 0):.1f}")
    logger.info("-" * 70)

    # Save training summary
    summary = {
        'total_timesteps': total_timesteps,
        'total_episodes': len(training_callback.episode_rewards),
        'final_mean_reward': eval_results['mean_reward'],
        'final_mean_pnl': eval_results['mean_pnl'],
        'final_win_rate': eval_results.get('win_rate', 0),
        'action_distribution': training_callback.action_counts,
        'avg_episode_length': float(np.mean(training_callback.episode_lengths)) if training_callback.episode_lengths else 0
    }

    summary_path = log_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary saved to: {summary_path}")

    # v36 OOD FIX: Save training data statistics for runtime OOD detection
    # These stats enable comparison of live/OOS data against training distribution
    training_stats = {
        'market_feat_mean': market_feat_mean.tolist(),
        'market_feat_std': market_feat_std.tolist(),
        'n_training_samples': len(train_data[3]),  # close_prices length
        'volatility_stats': {
            'atr_mean': float(market_feat_mean[0]) if len(market_feat_mean) > 0 else 0.0,
            'atr_std': float(market_feat_std[0]) if len(market_feat_std) > 0 else 1.0,
        },
        'augmentation_applied': use_vol_augmentation,
        'augmentation_ratio': vol_augment_ratio if use_vol_augmentation else 0.0,
    }

    stats_path = log_dir / 'training_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    logger.info(f"Training statistics for OOD detection saved to: {stats_path}")

    # v37 OOD FIX: Save comprehensive training baseline for training-anchored detection
    # Unlike v36 rolling features, this baseline is FIXED and NEVER adapts to new data
    # NOTE: run_pipeline.py now saves this during feature engineering (two-pass approach)
    # This is a FALLBACK for direct train_agent.py usage without the full pipeline
    try:
        from src.data.ood_features import TrainingBaseline

        baseline_path = log_dir / 'training_baseline.json'
        if baseline_path.exists():
            logger.info(f"v37 TrainingBaseline already exists at: {baseline_path} (created by pipeline)")
        else:
            # Fallback: Create baseline if running train_agent.py directly
            train_df_5m = df_5m.iloc[start_idx:start_idx + split_idx]
            training_baseline = TrainingBaseline.from_training_data(train_df_5m)
            training_baseline.save(baseline_path)
            logger.info(f"v37 TrainingBaseline saved to: {baseline_path}")
    except Exception as e:
        logger.warning(f"Failed to save training baseline: {e}")

    # Cleanup
    train_env.close()
    eval_env.close()
    gc.collect()

    return agent, training_info


def load_and_evaluate(
    agent_path: str,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_45m: pd.DataFrame,
    feature_cols: list,
    analyst_path: str,
    device: Optional[torch.device] = None,
    n_episodes: int = 50,
    config: Optional[object] = None,
) -> Dict:
    """
    Load a trained agent and evaluate it.

    Args:
        agent_path: Path to saved agent
        df_5m: 5-minute DataFrame (base timeframe)
        df_15m: 15-minute DataFrame (medium timeframe)
        df_45m: 45-minute DataFrame (trend timeframe)
        feature_cols: Feature columns
        analyst_path: Path to analyst model
        device: Torch device
        n_episodes: Number of evaluation episodes
        config: Configuration object (for date-based splits)

    Returns:
        Evaluation results
    """
    if device is None:
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    logger.info(f"Loading agent from {agent_path}")
    logger.info(f"Evaluating on {n_episodes} episodes")

    # Load analyst
    feature_dims = {'5m': len(feature_cols), '15m': len(feature_cols), '45m': len(feature_cols)}
    analyst = load_analyst(analyst_path, feature_dims, device, freeze=True)

    # Prepare data (use last portion as test)
    data_5m, data_15m, data_45m, close_prices, market_features, returns, rolling_lookback_data = prepare_env_data(
        df_5m, df_15m, df_45m, feature_cols
    )

    # v38 FIX: Use date-based split consistent with training
    n_total = len(close_prices)
    if config and hasattr(config, 'data_split') and config.data_split.use_date_splits:
        lookback_5m = config.analyst.lookback_5m
        lookback_15m = config.analyst.lookback_15m
        lookback_45m = config.analyst.lookback_45m
        subsample_15m, subsample_45m = 3, 9
        start_idx = max(
            lookback_5m,
            (lookback_15m - 1) * subsample_15m + 1,
            (lookback_45m - 1) * subsample_45m + 1,
        )
        train_end_date = pd.Timestamp(config.data_split.train_end_date)
        trimmed_index = df_5m.index[start_idx:start_idx + n_total]
        train_mask = trimmed_index < train_end_date
        test_start = int(train_mask.sum())
        logger.info(f"Date-based eval split: test starts at index {test_start} ({config.data_split.train_end_date})")
    else:
        test_start = int(0.70 * n_total)
    # Create test environment
    test_env = create_trading_env(
        data_5m=data_5m[test_start:],
        data_15m=data_15m[test_start:],
        data_45m=data_45m[test_start:],
        close_prices=close_prices[test_start:],
        market_features=market_features[test_start:],
        returns=returns[test_start:] if returns is not None else None,
        analyst_model=analyst,
        device=device,
    )
    test_env = Monitor(test_env)

    # Load agent
    agent = SniperAgent.load(agent_path, test_env, device='cpu')  # SB3 more stable on CPU

    # Evaluate
    logger.info("Running evaluation...")
    results = agent.evaluate(test_env, n_episodes=n_episodes)

    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS:")
    logger.info(f"  Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    logger.info(f"  Mean PnL: {results['mean_pnl']:.2f} pips")
    logger.info(f"  Win Rate: {results.get('win_rate', 0)*100:.1f}%")
    logger.info("=" * 70)

    test_env.close()
    return results


if __name__ == '__main__':
    logger.info("Use this module via: python -m src.training.train_agent")
    logger.info("Or import and call train_agent() function")
