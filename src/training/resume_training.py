"""
Agent Finetuning Script - Refine a pretrained PPO agent with easier conditions.

This script is for finetuning agent checkpoints that performed well during main training.
The analyst model remains frozen and unchanged.

Refactored to align with `src/training/train_agent.py` for consistent data processing
and Analyst cache usage.

Key differences from main training ("super hard mode"):
- Lower spread (5 pips vs 50) - easier execution costs
- Lower learning rate (5e-5 vs 3e-4) - stable refinement
- Lower entropy (0.005 vs 0.02) - exploitation over exploration

Saves to models/agent/finetune/ to keep separate from main training.
Uses the same training data split (70/30 - OOS test data untouched).
"""

import os
import sys
import logging
import argparse
import shutil
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from config.settings import Config
# Import shared logic from main training script
from src.training.train_agent import prepare_env_data, create_trading_env
from src.training.precompute_analyst import load_cached_analyst_outputs
from src.environments.env_factory import make_vec_env, prepare_env_kwargs_for_vectorization
from src.models.analyst import load_analyst
from src.agents.sniper_agent import SniperAgent
from src.utils.callbacks import AgentTrainingLogger

# ============= FINETUNING CONFIGURATION =============
FINETUNE_CONFIG = {
    # Easier environment (relaxed from "super hard mode")
    'spread_pips': 2.0,           # Was 50 in main training
    'slippage_pips': 0.0,         # Keep at 0

    # Conservative learning (stable refinement)
    'learning_rate': 5e-5,        # 6x lower than main training

    # Exploitation-focused (minimal exploration)
    'ent_coef': 0.005,            # 4x lower than initial

    # Training duration
    'total_timesteps': 100_000_000,  # 100M steps (adjustable via CLI)
    # Paths
    'save_dir': 'models/agent/finetune',
    'log_file': 'finetune_training.log',
}
# ====================================================


class FixedEntropyCallback(BaseCallback):
    """Force a constant entropy coefficient during finetuning."""

    def __init__(self, ent_coef: float, verbose: int = 0):
        super().__init__(verbose)
        self.ent_coef = ent_coef

    def _on_step(self) -> bool:
        self.model.ent_coef = self.ent_coef
        return True


def parse_args():
    """Parse command-line arguments for finetuning customization."""
    parser = argparse.ArgumentParser(
        description='Finetune a pretrained PPO agent with relaxed trading conditions'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint to finetune (default: models/agent/final_model.zip)'
    )
    parser.add_argument(
        '--spread', type=float, default=FINETUNE_CONFIG['spread_pips'],
        help=f"Spread in pips (default: {FINETUNE_CONFIG['spread_pips']})"
    )
    parser.add_argument(
        '--lr', type=float, default=FINETUNE_CONFIG['learning_rate'],
        help=f"Learning rate (default: {FINETUNE_CONFIG['learning_rate']})"
    )
    parser.add_argument(
        '--ent-coef', type=float, default=FINETUNE_CONFIG['ent_coef'],
        help=f"Entropy coefficient (default: {FINETUNE_CONFIG['ent_coef']})"
    )
    parser.add_argument(
        '--timesteps', type=int, default=FINETUNE_CONFIG['total_timesteps'],
        help=f"Total timesteps (default: {FINETUNE_CONFIG['total_timesteps']})"
    )
    parser.add_argument(
        '--num-envs', type=int, default=None,
        help='Number of parallel environments (default: uses config.agent.n_envs)'
    )
    parser.add_argument(
        '--use-oos', action='store_true',
        help='Finetune on OOS data (last 30%%) instead of training data (first 70%%)'
    )
    return parser.parse_args()


def finetune_agent():
    """
    Finetune a pretrained PPO agent with relaxed trading conditions.
    Using shared workflow components from train_agent.py.
    """
    # Parse CLI arguments
    args = parse_args()

    # Create finetune directory (auto-clean on each run)
    finetune_dir = project_root / FINETUNE_CONFIG['save_dir']
    cleaned = False
    if finetune_dir.exists():
        shutil.rmtree(finetune_dir)
        cleaned = True
    finetune_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_path = finetune_dir / FINETUNE_CONFIG['log_file']
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path)
        ]
    )
    global logger
    logger = logging.getLogger('finetune_agent')
    if cleaned:
        logger.info(f"Cleared existing finetune artifacts in {finetune_dir}")

    config = Config()
    device = torch.device("cpu")  # Force CPU for PPO stability

    logger.info("=" * 60)
    logger.info("AGENT FINETUNING - Relaxed Conditions (Shared Architecture)")
    logger.info("=" * 60)
    logger.info(f"Using device: {device}")
    logger.info("Finetuning parameters:")
    logger.info(f"  Spread: {args.spread} pips (main training: {config.trading.spread_pips})")
    logger.info(f"  Learning rate: {args.lr} (main training: 3e-4)")
    logger.info(f"  Entropy coef: {args.ent_coef} (main training: 0.02)")
    logger.info(f"  min_hold_bars: {config.trading.min_hold_bars} (inherited from config)")
    logger.info(f"  SL/TP: {config.trading.sl_atr_multiplier}x / {config.trading.tp_atr_multiplier}x ATR (1:{config.trading.tp_atr_multiplier/config.trading.sl_atr_multiplier:.0f} R/R)")
    logger.info(f"  Total timesteps: {args.timesteps:,}")
    logger.info(f"  Save directory: {finetune_dir}")
    logger.info("=" * 60)

    # Determine n_envs from CLI or config
    n_envs = args.num_envs if args.num_envs is not None else config.agent.n_envs
    use_subproc = config.agent.use_subproc
    logger.info(f"Using {n_envs} parallel environment(s) (use_subproc={use_subproc})")

    # OVERRIDE CONFIG for Fine-tuning parameters
    # This ensures create_trading_env picks up the right values
    config.trading.spread_pips = args.spread
    config.trading.slippage_pips = FINETUNE_CONFIG['slippage_pips']
    config.agent.learning_rate = args.lr
    config.agent.ent_coef = args.ent_coef
    # Finetuning: keep mostly PnL rewards with light shaping penalties only.
    # Small opportunity cost penalty to discourage staying flat/wrong-direction.
    config.trading.opportunity_cost_multiplier = 0.1
    config.trading.opportunity_cost_cap = 0.02
    # Stronger underwater penalty to discourage holding deep losers.
    config.trading.underwater_penalty_coef = 0.1
    config.trading.underwater_threshold_atr = 1.0
    # Lower per-trade risk for finetuning.
    config.trading.risk_per_trade = 50.0
    config.trading.trade_entry_bonus = 0.0
    config.trading.use_alpha_reward = False
    config.trading.profit_scaling = config.trading.reward_scaling
    config.trading.loss_scaling = config.trading.reward_scaling

    # 1. Load Pre-processed Data
    data_processed_path = config.paths.data_processed
    logger.info(f"Loading processed data from {data_processed_path}...")

    try:
        df_5m = pd.read_parquet(data_processed_path / 'features_5m_normalized.parquet')
        df_15m = pd.read_parquet(data_processed_path / 'features_15m_normalized.parquet')
        df_45m = pd.read_parquet(data_processed_path / 'features_45m_normalized.parquet')
    except Exception as e:
        logger.error(f"Failed to load parquet files: {e}")
        return

    # 2. Define Feature Columns (must match prediction model)
    # Using bridge constants to ensure alignment
    from src.live.bridge_constants import MODEL_FEATURE_COLS
    feature_cols = list(MODEL_FEATURE_COLS)
    required_feature_cols = list(MODEL_FEATURE_COLS)

    if feature_cols != required_feature_cols:
        logger.warning("Overriding `feature_cols` with canonical MODEL_FEATURE_COLS ordering.")
        feature_cols = required_feature_cols

    # Load Analyst (same as main training)
    use_analyst = getattr(config.trading, 'use_analyst', True)
    if use_analyst:
        analyst_path = config.paths.models_analyst / 'best.pt'
        fallback_path = config.paths.models_analyst / 'best_model.pt'
        if not analyst_path.exists() and fallback_path.exists():
            analyst_path = fallback_path
        feature_dims = {'5m': len(feature_cols), '15m': len(feature_cols), '45m': len(feature_cols)}
        analyst = load_analyst(str(analyst_path), feature_dims, device, freeze=True)
        logger.info(f"Analyst loaded and frozen from {analyst_path}")
    else:
        logger.info("ANALYST DISABLED (use_analyst=False) - using market features only")
        analyst = None

    # 3. Augment Data (Sessions, Structure) using train_agent logic
    from src.data.features import (
        add_market_sessions,
        detect_fractals,
        detect_structure_breaks,
    )

    logger.info("Adding augmented features (sessions & structure)...")
    df_5m = add_market_sessions(df_5m)
    df_15m = add_market_sessions(df_15m)
    df_45m = add_market_sessions(df_45m)
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

    # 4. Prepare Environment Data (Windowing)
    lookback_5m = config.analyst.lookback_5m
    lookback_15m = config.analyst.lookback_15m
    lookback_45m = config.analyst.lookback_45m

    logger.info("Preparing environment data via `prepare_env_data`...")
    data_5m_arr, data_15m_arr, data_45m_arr, close_prices, market_features, returns, rolling_lookback_data = prepare_env_data(
        df_5m, df_15m, df_45m, feature_cols,
        lookback_5m, lookback_15m, lookback_45m
    )

    # 5. Split into train/eval (train portion only for finetune)
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

    if isinstance(df_5m.index, pd.DatetimeIndex) or hasattr(df_5m.index, 'to_pydatetime'):
        try:
            timestamps = (df_5m.index[start_idx:start_idx + n_samples].astype('int64') // 10**9).values
        except Exception as e:
            logger.warning(f"Failed to extract timestamps: {e}")

    # v38 FIX: Use date-based split from config to match train_agent.py and pipeline
    n_total = len(close_prices)
    if config and hasattr(config, 'data_split') and config.data_split.use_date_splits:
        train_end_date = pd.Timestamp(config.data_split.train_end_date)
        trimmed_index = df_5m.index[start_idx:start_idx + n_total]
        train_mask = trimmed_index < train_end_date
        split_idx = int(train_mask.sum())
        logger.info(f"Date-based split: train ends at index {split_idx} ({config.data_split.train_end_date})")
    else:
        split_idx = int(0.70 * n_total)
        logger.info(f"Percentage-based split: train ends at index {split_idx} (70%)")
    
    # Determine which portion of data to use
    if args.use_oos:
        logger.info(">>> USING OOS DATA FOR FINETUNING (excluding training data) <<<")
        train_start_idx = split_idx
        train_end_idx = len(close_prices)
    else:
        train_start_idx = 0
        train_end_idx = split_idx
    logger.info(f"Finetune samples: {train_end_idx - train_start_idx} (indices {train_start_idx}:{train_end_idx})")

    train_data = (
        data_5m_arr[train_start_idx:train_end_idx],
        data_15m_arr[train_start_idx:train_end_idx],
        data_45m_arr[train_start_idx:train_end_idx],
        close_prices[train_start_idx:train_end_idx],
        market_features[train_start_idx:train_end_idx],
        returns[train_start_idx:train_end_idx] if returns is not None else None,
    )
    train_ohlc = ohlc_data[train_start_idx:train_end_idx] if ohlc_data is not None else None
    train_timestamps = timestamps[train_start_idx:train_end_idx] if timestamps is not None else None

    # Live-style rolling normalization warmup for training
    rolling_warmup_train = rolling_lookback_data

    # Compute market feature normalization stats from TRAINING data only
    train_market_features = train_data[4]
    market_feat_mean = train_market_features.mean(axis=0).astype(np.float32)
    market_feat_std = train_market_features.std(axis=0).astype(np.float32)
    market_feat_std = np.where(market_feat_std > 1e-8, market_feat_std, 1.0).astype(np.float32)

    logger.info("Market feature normalization stats (from training data):")
    logger.info(f"  Mean: {market_feat_mean}")
    logger.info(f"  Std:  {market_feat_std}")

    max_steps_per_episode = getattr(config.trading, "max_steps_per_episode", 500)

    # 6. Load Analyst Cache (if available)
    analyst_cache_path = data_processed_path / 'analyst_cache.npz'
    train_analyst_cache = None

    if use_analyst and analyst_cache_path.exists():
        logger.info(f"Loading pre-computed Analyst cache from {analyst_cache_path}")
        try:
            full_cache = load_cached_analyst_outputs(str(analyst_cache_path))

            expected_samples = len(close_prices)
            cache_samples = len(full_cache['contexts'])
            if cache_samples != expected_samples:
                logger.warning(
                    f"Analyst cache mismatch: cache has {cache_samples} samples vs expected {expected_samples}. "
                    f"Cache lookbacks=({full_cache['lookback_5m']},{full_cache['lookback_15m']},{full_cache['lookback_45m']}), "
                    f"Config lookbacks=({lookback_5m},{lookback_15m},{lookback_45m}). "
                    "Regenerate cache with: python src/training/precompute_analyst.py"
                )
                logger.warning("Ignoring misaligned cache - falling back to on-the-fly Analyst inference")
            else:
                train_analyst_cache = {
                    'contexts': full_cache['contexts'][train_start_idx:train_end_idx],
                    'probs': full_cache['probs'][train_start_idx:train_end_idx],
                    'activations_15m': None,
                    'activations_1h': None,
                    'activations_4h': None,
                }
                logger.info(f"Using sequential Analyst context: train={len(train_analyst_cache['contexts'])}")
        except Exception as e:
            logger.warning(f"Failed to load Analyst cache: {e}")
            logger.info("Falling back to standard precomputation")

    # 7. Create Training Environment (vectorized if n_envs > 1)
    if n_envs > 1:
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
            timestamps=train_timestamps,
            use_analyst=use_analyst,
            rolling_lookback_data=rolling_warmup_train,
        )

        # Create vectorized environment
        train_env = make_vec_env(
            n_envs=n_envs,
            use_subproc=use_subproc,
            env_kwargs=train_env_kwargs,
            seed=config.seed,
        )

        logger.info(f"Vectorized training environment created: {type(train_env).__name__} with {n_envs} envs")
        logger.info(f"Observation space: {train_env.observation_space}")
        logger.info(f"Action space: {train_env.action_space}")
        logger.info(f"Training with {n_envs} parallel environments (effective batch: {n_envs * config.agent.n_steps} steps/update)")
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
            timestamps=train_timestamps,
            use_analyst=use_analyst,
            rolling_lookback_data=rolling_warmup_train,
        )

        logger.info(f"Observation space: {train_env.observation_space}")
        logger.info(f"Action space: {train_env.action_space}")

        train_env = Monitor(train_env)

    # 8. Load Agent Checkpoint
    if args.checkpoint:
        model_path = args.checkpoint
    else:
        model_path = config.paths.models_agent / 'final_model.zip'

    if not os.path.exists(model_path):
        logger.error(f"Checkpoint not found: {model_path}")
        return

    logger.info(f"Loading agent checkpoint from {model_path}...")
    agent = SniperAgent.load(model_path, env=train_env, device=device)

    # 9. Apply Finetuning Overrides
    agent.model.learning_rate = args.lr
    agent.model.lr_schedule = get_schedule_fn(args.lr)
    agent.model.ent_coef = args.ent_coef

    try:
        agent.model._update_learning_rate(agent.model.optimizer)
    except Exception:
        pass

    logger.info("Overridden agent parameters:")
    logger.info(f"  learning_rate: {args.lr}")
    logger.info(f"  ent_coef: {args.ent_coef}")

    training_logger = AgentTrainingLogger(
        log_dir=str(finetune_dir),
        log_freq=5000,
        verbose=1
    )
    fixed_entropy = FixedEntropyCallback(args.ent_coef)

    # 10. Run Finetuning
    logger.info(f"Starting finetuning for {args.timesteps:,} steps...")

    try:
        agent.train(
            total_timesteps=args.timesteps,
            save_path=str(finetune_dir),
            checkpoint_save_freq=100_000,
            callbacks=[fixed_entropy],
            callback=training_logger,
            reset_num_timesteps=False
        )

        final_model_path = finetune_dir / 'finetuned_model'
        agent.save(final_model_path)
        logger.info(f"Finetuning complete! Model saved to {final_model_path}.zip")

    except KeyboardInterrupt:
        logger.info("Finetuning interrupted. Saving current model...")
        agent.save(finetune_dir / 'interrupted_model')


if __name__ == "__main__":
    finetune_agent()
