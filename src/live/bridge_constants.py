"""
Constants shared by the MT5 live bridge.

Keep this module lightweight: no heavy imports (SB3/matplotlib/etc).
"""

from __future__ import annotations

# Ordered Analyst/Agent model input features (per timeframe).
# IMPORTANT: Ordering matters and must match the ordering used during training.
MODEL_FEATURE_COLS: list[str] = [
    "returns",
    "volatility",
    "sma_distance",
    "ema_gap",
    "ema_acceleration",
    "ema_crossover_recency",
    "dist_to_resistance",
    "dist_to_support",
    "sr_strength_r",
    "sr_strength_s",
    "session_asian",
    "session_london",
    "session_ny",
    # v2 Structure Features (continuous, RL-friendly)
    "structure_fade",
    "bars_since_bos",
    "bars_since_choch",
    "bos_magnitude",
    "choch_magnitude",
    "bos_streak",
    "atr_context",
    # v35 FIX: Added regime-robust percentile features
    "atr_percentile",
    "chop_percentile",
    "sma_distance_percentile",
    "volatility_percentile",
    # v37 FIX: Training-anchored OOD features (FIXED stats, never adapt)
    "volatility_vs_training",
    "returns_skew_shift",
    "atr_vs_training",
    "ood_score",
    # Mean Reversion Features (4)
    "bb_percent_b",
    "bb_bandwidth",
    "williams_r",
    "rsi",
    "rsi_divergence",
]

# Market feature columns used in the RL observation.
MARKET_FEATURE_COLS: list[str] = [
    "atr",
    "chop",
    "adx",
    "sma_distance",
    "ema_gap",
    "ema_acceleration",
    "ema_crossover_recency",
    "dist_to_support",
    "dist_to_resistance",
    "sr_strength_r",
    "sr_strength_s",
    "session_asian",
    "session_london",
    "session_ny",
    # v2 Structure Features (continuous, RL-friendly)
    "structure_fade",
    "bars_since_bos",
    "bars_since_choch",
    "bos_magnitude",
    "choch_magnitude",
    "bos_streak",
    # Extra model features so PPO can learn without Analyst context.
    "returns",
    "volatility",
    "atr_context",
    # v35 FIX: Added regime-robust percentile features
    "atr_percentile",
    "chop_percentile",
    "sma_distance_percentile",
    "volatility_percentile",
    # v37 FIX: Training-anchored OOD features (FIXED stats, never adapt)
    "volatility_vs_training",
    "returns_skew_shift",
    "atr_vs_training",
    "ood_score",
    # Mean Reversion Features (4)
    "bb_percent_b",
    "bb_bandwidth",
    "williams_r",
    "rsi",
    "rsi_divergence",
]

# TradingEnv size mapping (must match TradingEnv.POSITION_SIZES)
# CRITICAL: Must match exactly or agent will take wrong-sized positions in live trading!
POSITION_SIZES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)  # v27: Match TradingEnv
