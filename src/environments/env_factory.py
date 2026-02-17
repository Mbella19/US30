"""
Environment factory for vectorized training.

This module provides picklable environment creation functions for use with
SubprocVecEnv (multiprocessing) or DummyVecEnv (single process).

Key design decisions:
- NO PyTorch models passed to subprocesses (not picklable)
- Use precomputed_analyst_cache (numpy arrays) instead of analyst_model
- All parameters must be picklable (numpy, dicts, primitives)
"""

import logging
from typing import Dict, Any, Optional, Union, Callable

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.environments.trading_env import TradingEnv

logger = logging.getLogger(__name__)


def make_env_fn(
    rank: int,
    seed: int,
    env_kwargs: Dict[str, Any],
) -> Callable[[], TradingEnv]:
    """
    Create a callable that returns a TradingEnv instance.

    This function returns a closure that can be pickled and sent to subprocesses.
    The closure creates the environment when called.

    Args:
        rank: Index of this environment (0 to n_envs-1)
        seed: Base random seed (will be offset by rank)
        env_kwargs: Dictionary of arguments to pass to TradingEnv

    Returns:
        Callable that creates a TradingEnv wrapped in Monitor
    """
    def _init() -> TradingEnv:
        # Set seed for this specific environment
        env_seed = seed + rank

        # Create the environment
        env = TradingEnv(
            **env_kwargs,
            # Don't pass analyst_model - use precomputed cache only
            analyst_model=None,
            # Device not needed when using precomputed cache
            device=None,
        )

        # Seed the environment
        env.reset(seed=env_seed)

        # Wrap in Monitor for episode stats
        env = Monitor(env)

        return env

    return _init


def make_vec_env(
    n_envs: int,
    use_subproc: bool,
    env_kwargs: Dict[str, Any],
    seed: int = 42,
) -> Union[SubprocVecEnv, DummyVecEnv]:
    """
    Create a vectorized environment for parallel training.

    Args:
        n_envs: Number of parallel environments
        use_subproc: If True, use SubprocVecEnv (multiprocessing).
                     If False, use DummyVecEnv (single process).
        env_kwargs: Dictionary of arguments to pass to each TradingEnv.
                    Must NOT contain unpicklable objects (nn.Module, torch.device).
                    Should contain precomputed_analyst_cache instead of analyst_model.
        seed: Base random seed (each env gets seed + rank)

    Returns:
        Vectorized environment (SubprocVecEnv or DummyVecEnv)

    Note:
        For SubprocVecEnv, each subprocess gets a copy of env_kwargs.
        Memory usage: ~33MB per environment (data arrays + analyst cache).
        8 environments = ~264MB additional memory.
    """
    # Validate env_kwargs doesn't contain unpicklable objects
    if 'analyst_model' in env_kwargs and env_kwargs['analyst_model'] is not None:
        logger.warning(
            "analyst_model passed to make_vec_env will be ignored. "
            "Use precomputed_analyst_cache instead for vectorized environments."
        )
        env_kwargs = {k: v for k, v in env_kwargs.items() if k != 'analyst_model'}

    if 'device' in env_kwargs:
        # Remove device - not needed with precomputed cache
        env_kwargs = {k: v for k, v in env_kwargs.items() if k != 'device'}

    # Create environment factories
    env_fns = [
        make_env_fn(rank=i, seed=seed, env_kwargs=env_kwargs)
        for i in range(n_envs)
    ]

    if use_subproc and n_envs > 1:
        logger.info(f"Creating SubprocVecEnv with {n_envs} parallel environments...")
        # SubprocVecEnv spawns separate Python processes
        # Each process has its own GIL, enabling true parallelism
        vec_env = SubprocVecEnv(env_fns, start_method='spawn')
    else:
        logger.info(f"Creating DummyVecEnv with {n_envs} environments...")
        # DummyVecEnv runs all envs in the main process (no parallelism)
        # Still useful for batched stepping and consistent API
        vec_env = DummyVecEnv(env_fns)

    return vec_env


def prepare_env_kwargs_for_vectorization(
    data_5m: np.ndarray,
    data_15m: np.ndarray,
    data_45m: np.ndarray,
    close_prices: np.ndarray,
    market_features: np.ndarray,
    returns: Optional[np.ndarray],
    config: Any,
    market_feat_mean: Optional[np.ndarray],
    market_feat_std: Optional[np.ndarray],
    regime_labels: Optional[np.ndarray],
    use_regime_sampling: bool,
    precomputed_analyst_cache: Optional[Dict[str, np.ndarray]],
    ohlc_data: Optional[np.ndarray],
    timestamps: Optional[np.ndarray],
    use_analyst: bool,
    rolling_lookback_data: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    Prepare environment kwargs dictionary for vectorized environment creation.

    This extracts all necessary values from config and creates a picklable
    dictionary that can be passed to make_vec_env.

    Args:
        data_5m: 5-minute window data
        data_15m: 15-minute window data
        data_45m: 45-minute window data
        close_prices: Close prices for PnL calculation
        market_features: Market features for observations
        returns: Returns data for agent lookback
        config: Configuration object (values extracted, object not pickled)
        market_feat_mean: Mean for market feature normalization
        market_feat_std: Std for market feature normalization
        regime_labels: Regime labels for balanced sampling
        use_regime_sampling: Whether to use regime-balanced episode starts
        precomputed_analyst_cache: Pre-computed analyst outputs (contexts, probs)
        ohlc_data: OHLC data for visualization
        timestamps: Timestamps for visualization
        use_analyst: Whether to use analyst context in observations
        rolling_lookback_data: Data for rolling normalization warmup

    Returns:
        Dictionary of env kwargs that can be safely pickled
    """
    # Extract trading config values
    trading_cfg = getattr(config, 'trading', config) if config else None

    # Default values matching config/settings.py TradingConfig
    spread_pips = 10.0              # config.trading.spread_pips
    slippage_pips = 0.0             # config.trading.slippage_pips
    fomo_penalty = 0.0
    chop_penalty = 0.0
    fomo_threshold_atr = 4.0        # config.trading.fomo_threshold_atr
    fomo_lookback_bars = 24         # config.trading.fomo_lookback_bars
    chop_threshold = 80.0           # config.trading.chop_threshold
    max_steps = 500
    reward_scaling = 0.01           # config.trading.reward_scaling
    context_dim = 32                # config.analyst.context_dim
    trade_entry_bonus = 0.1         # config.trading.trade_entry_bonus
    holding_bonus = 0.0             # config.trading.holding_bonus (DEPRECATED)
    noise_level = 0.05
    profit_scaling = 0.01           # config.trading.profit_scaling
    loss_scaling = 0.01             # config.trading.loss_scaling
    use_alpha_reward = True         # config.trading.use_alpha_reward
    alpha_baseline_exposure = 0.7   # config.trading.alpha_baseline_exposure
    sl_atr_multiplier = 2.0         # config.trading.sl_atr_multiplier
    tp_atr_multiplier = 6.0         # config.trading.tp_atr_multiplier
    use_stop_loss = True            # config.trading.use_stop_loss
    use_take_profit = True          # config.trading.use_take_profit
    volatility_sizing = True
    risk_per_trade = 100.0          # config.trading.risk_per_trade
    num_classes = 2
    enforce_analyst_alignment = False  # config.trading.enforce_analyst_alignment
    use_sparse_rewards = False      # config.trading.use_sparse_rewards (DEPRECATED)
    loss_tolerance_atr = 1.0        # config.trading.loss_tolerance_atr
    min_hold_bars = 0               # config.trading.min_hold_bars
    early_exit_profit_atr = 0.0     # config.trading.early_exit_profit_atr
    break_even_atr = 2.0            # config.trading.break_even_atr
    opportunity_cost_multiplier = 0.0  # config.trading.opportunity_cost_multiplier
    opportunity_cost_cap = 0.2      # config.trading.opportunity_cost_cap
    rolling_norm_min_samples = 1
    agent_lookback_window = 6

    # Override from config if available
    if trading_cfg is not None:
        spread_pips = getattr(trading_cfg, 'spread_pips', spread_pips)
        slippage_pips = getattr(trading_cfg, 'slippage_pips', slippage_pips)
        fomo_penalty = getattr(trading_cfg, 'fomo_penalty', fomo_penalty)
        chop_penalty = getattr(trading_cfg, 'chop_penalty', chop_penalty)
        fomo_threshold_atr = getattr(trading_cfg, 'fomo_threshold_atr', fomo_threshold_atr)
        fomo_lookback_bars = getattr(trading_cfg, 'fomo_lookback_bars', fomo_lookback_bars)
        chop_threshold = getattr(trading_cfg, 'chop_threshold', chop_threshold)
        max_steps = getattr(trading_cfg, 'max_steps_per_episode', max_steps)
        reward_scaling = getattr(trading_cfg, 'reward_scaling', reward_scaling)
        sl_atr_multiplier = getattr(trading_cfg, 'sl_atr_multiplier', sl_atr_multiplier)
        tp_atr_multiplier = getattr(trading_cfg, 'tp_atr_multiplier', tp_atr_multiplier)
        use_stop_loss = getattr(trading_cfg, 'use_stop_loss', use_stop_loss)
        use_take_profit = getattr(trading_cfg, 'use_take_profit', use_take_profit)
        enforce_analyst_alignment = getattr(trading_cfg, 'enforce_analyst_alignment', enforce_analyst_alignment)
        trade_entry_bonus = getattr(trading_cfg, 'trade_entry_bonus', trade_entry_bonus)
        holding_bonus = getattr(trading_cfg, 'holding_bonus', holding_bonus)
        noise_level = getattr(trading_cfg, 'noise_level', noise_level)
        profit_scaling = getattr(trading_cfg, 'profit_scaling', profit_scaling)
        loss_scaling = getattr(trading_cfg, 'loss_scaling', loss_scaling)
        use_alpha_reward = getattr(trading_cfg, 'use_alpha_reward', use_alpha_reward)
        alpha_baseline_exposure = getattr(trading_cfg, 'alpha_baseline_exposure', alpha_baseline_exposure)
        use_sparse_rewards = getattr(trading_cfg, 'use_sparse_rewards', use_sparse_rewards)
        loss_tolerance_atr = getattr(trading_cfg, 'loss_tolerance_atr', loss_tolerance_atr)
        min_hold_bars = getattr(trading_cfg, 'min_hold_bars', min_hold_bars)
        early_exit_profit_atr = getattr(trading_cfg, 'early_exit_profit_atr', early_exit_profit_atr)
        break_even_atr = getattr(trading_cfg, 'break_even_atr', break_even_atr)
        opportunity_cost_multiplier = getattr(trading_cfg, 'opportunity_cost_multiplier', opportunity_cost_multiplier)
        opportunity_cost_cap = getattr(trading_cfg, 'opportunity_cost_cap', opportunity_cost_cap)
        rolling_norm_min_samples = getattr(trading_cfg, 'rolling_norm_min_samples', rolling_norm_min_samples)
        agent_lookback_window = getattr(trading_cfg, 'agent_lookback_window', agent_lookback_window)

    # Get context_dim and num_classes from analyst cache if available
    if precomputed_analyst_cache is not None:
        contexts = precomputed_analyst_cache.get('contexts')
        probs = precomputed_analyst_cache.get('probs')
        if contexts is not None:
            context_dim = contexts.shape[1]
        if probs is not None:
            num_classes = probs.shape[1]

    return {
        'data_5m': data_5m,
        'data_15m': data_15m,
        'data_45m': data_45m,
        'close_prices': close_prices,
        'market_features': market_features,
        'context_dim': context_dim,
        'spread_pips': spread_pips,
        'slippage_pips': slippage_pips,
        'fomo_penalty': fomo_penalty,
        'chop_penalty': chop_penalty,
        'fomo_threshold_atr': fomo_threshold_atr,
        'fomo_lookback_bars': fomo_lookback_bars,
        'chop_threshold': chop_threshold,
        'max_steps': max_steps,
        'reward_scaling': reward_scaling,
        'trade_entry_bonus': trade_entry_bonus,
        'holding_bonus': holding_bonus,
        'profit_scaling': profit_scaling,
        'loss_scaling': loss_scaling,
        'use_alpha_reward': use_alpha_reward,
        'alpha_baseline_exposure': alpha_baseline_exposure,
        'noise_level': noise_level,
        'market_feat_mean': market_feat_mean,
        'market_feat_std': market_feat_std,
        'sl_atr_multiplier': sl_atr_multiplier,
        'tp_atr_multiplier': tp_atr_multiplier,
        'use_stop_loss': use_stop_loss,
        'use_take_profit': use_take_profit,
        'regime_labels': regime_labels,
        'use_regime_sampling': use_regime_sampling,
        'volatility_sizing': volatility_sizing,
        'risk_per_trade': risk_per_trade,
        'num_classes': num_classes,
        'enforce_analyst_alignment': enforce_analyst_alignment,
        'use_sparse_rewards': use_sparse_rewards,
        'loss_tolerance_atr': loss_tolerance_atr,
        'min_hold_bars': min_hold_bars,
        'early_exit_profit_atr': early_exit_profit_atr,
        'break_even_atr': break_even_atr,
        'opportunity_cost_multiplier': opportunity_cost_multiplier,
        'opportunity_cost_cap': opportunity_cost_cap,
        'precomputed_analyst_cache': precomputed_analyst_cache,
        'ohlc_data': ohlc_data,
        'timestamps': timestamps,
        'returns': returns,
        'agent_lookback_window': agent_lookback_window,
        'use_analyst': use_analyst,
        'rolling_lookback_data': rolling_lookback_data,
        'rolling_norm_min_samples': rolling_norm_min_samples,
    }
