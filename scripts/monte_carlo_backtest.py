#!/usr/bin/env python3
"""
Monte Carlo Backtest on Full Training Data
==========================================

Performs a Monte Carlo simulation by running multiple backtest episodes
starting at random points within the full training dataset.

Key Features:
- Loads full Oanda training history
- Uses saved training normalizers (1:1 parity)
- Runs N randomized episodes (Block Bootstrapping)
- Zero spread/slippage as requested
- Aggregates performance metrics
"""

import sys
import os
from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config, get_device, clear_memory
from src.data.resampler import resample_all_timeframes, align_timeframes
from src.data.features import engineer_all_features
from src.data.normalizer import FeatureNormalizer
from src.models.analyst import load_analyst
from src.training.train_agent import prepare_env_data, create_trading_env
from src.agents.sniper_agent import SniperAgent
from src.evaluation.backtest import run_backtest, BacktestResult

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_process_full_data(config: Config):
    """Load and process the entire training dataset."""
    data_path = config.paths.training_data_dir / "US30_USD_1min_data.csv"
    logger.info(f"Loading full data from {data_path}...")
    
    # Load CSV
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()
    logger.info(f"Loaded {len(df):,} rows ({df.index.min()} to {df.index.max()})")
    
    # Resample
    logger.info("Resampling...")
    resampled = resample_all_timeframes(df, config.data.timeframes)
    df_5m, df_15m, df_45m = resampled['5m'], resampled['15m'], resampled['45m']
    
    # Feature Engineering
    logger.info("Engineering features (this may take a moment)...")
    feature_config = {k: getattr(config.features, k) for k in [
        'fractal_window', 'sr_lookback',
        'sma_period', 'chop_period', 'adx_period', 'atr_period'
    ]}
    
    df_5m = engineer_all_features(df_5m, feature_config)
    df_15m = engineer_all_features(df_15m, feature_config)
    df_45m = engineer_all_features(df_45m, feature_config)
    
    # Align
    df_5m, df_15m, df_45m = align_timeframes(df_5m, df_15m, df_45m)
    
    # Clean NaNs
    valid = ~df_5m.isna().any(axis=1) & ~df_15m.isna().any(axis=1) & ~df_45m.isna().any(axis=1)
    df_5m = df_5m[valid]
    df_15m = df_15m[valid]
    df_45m = df_45m[valid]
    
    # Normalize (Using SAVED Training Normalizers)
    logger.info("Normalizing with saved training stats...")
    for tf, df_tf in [('5m', df_5m), ('15m', df_15m), ('45m', df_45m)]:
        norm_path = config.paths.models_analyst / f'normalizer_{tf}.pkl'
        if norm_path.exists():
            norm = FeatureNormalizer.load(norm_path)
            if tf == '5m': df_5m = norm.transform(df_5m)
            elif tf == '15m': df_15m = norm.transform(df_15m)
            else: df_45m = norm.transform(df_45m)
        else:
            logger.error(f"Normalizer missing for {tf}!")
            sys.exit(1)
            
    return df_5m, df_15m, df_45m

def run_simulation(args):
    config = Config()
    device = torch.device('cpu') # CPU is sufficient for backtesting
    
    # Force Zero Costs
    logger.info("Forcing ZERO trading costs (Spread=0, Slippage=0)")
    config.trading.spread_pips = 0.0
    config.trading.slippage_pips = 0.0
    
    # Load Data
    df_5m, df_15m, df_45m = load_and_process_full_data(config)
    
    # Identify Features
    feature_cols = [
        'returns', 'volatility', 'sma_distance',
        'dist_to_resistance', 'dist_to_support',
        'sr_strength_r', 'sr_strength_s',
        'session_asian', 'session_london', 'session_ny',
        'structure_fade', 'bars_since_bos', 'bars_since_choch',
        'bos_magnitude', 'choch_magnitude',
        'atr_context'
    ]
    feature_cols = [c for c in feature_cols if c in df_5m.columns]
    
    # Prepare Environment Data Structure (Full)
    lookback_5m, lookback_15m, lookback_45m = 48, 16, 6
    data_5m, data_15m, data_45m, close_prices, market_features, returns, rolling_lookback = prepare_env_data(
        df_5m, df_15m, df_45m, feature_cols, lookback_5m, lookback_15m, lookback_45m
    )
    
    # OHLC & Timestamps
    start_idx = max(lookback_5m, (lookback_15m-1)*3+1, (lookback_45m-1)*9+1)
    n_samples = len(close_prices)
    timestamps = (df_5m.index[start_idx:start_idx+n_samples].astype('int64') // 10**9).values
    ohlc_data = df_5m[['open','high','low','close']].values[start_idx:start_idx+n_samples].astype(np.float32)
    
    # Load Market Stats
    stats = np.load(config.paths.models_agent / 'market_feat_stats.npz')
    mf_mean, mf_std = stats['mean'].astype(np.float32), stats['std'].astype(np.float32)
    mf_std = np.where(mf_std > 1e-8, mf_std, 1.0).astype(np.float32)
    
    # Load Analyst
    analyst_path = config.paths.models_analyst / 'best.pt'
    feature_dims = {'5m': len(feature_cols), '15m': len(feature_cols), '45m': len(feature_cols)}
    analyst = load_analyst(str(analyst_path), feature_dims, device, freeze=True)
    
    # Load Agent
    model_path = args.model
    # Create temp env just to load agent
    # Provide a generous buffer (equal size for all) to prevent mismatch errors
    logger.info("creating temp env for model loading...")
    temp_env = create_trading_env(
        data_5m[:2000], data_15m[:2000], data_45m[:2000], close_prices[:2000], market_features[:2000],
        analyst_model=analyst, config=config.trading, device=device,
        market_feat_mean=mf_mean, market_feat_std=mf_std, returns=returns[:2000],
        ohlc_data=ohlc_data[:2000], timestamps=timestamps[:2000], use_analyst=True
    )
    agent = SniperAgent.load(model_path, temp_env, device='cpu')
    logger.info(f"Loaded agent from {model_path}")
    
    # Monte Carlo Loop
    results_list = []
    equity_curves = []
    
    episode_length = args.length # e.g., 4000 bars (~2 weeks)
    num_episodes = args.episodes
    
    # Ensure we don't go out of bounds
    # Extra safety buffer for higher timeframes alignment (need ~100 bars)
    max_start = n_samples - episode_length - 200 
    
    if max_start <= 0:
        logger.error("Dataset too short for requested episode length!")
        sys.exit(1)
        
    logger.info(f"Starting Monte Carlo Simulation: {num_episodes} episodes x {episode_length} bars")
    
    for i in tqdm(range(num_episodes)):
        # Random start index
        rand_start = np.random.randint(0, max_start)
        
        s = rand_start
        e = rand_start + episode_length
        
        # Robust Slicing using Timestamps
        t_start = df_5m.index[s]
        t_end = df_5m.index[e-1] # Last timestamp in the 5m episode
        
        # Find corresponding indices in higher timeframes
        # We search in the DataFrame index which is datetime aligned
        
        # 15m
        idx_15_start = df_15m.index.searchsorted(t_start)
        idx_15_end = df_15m.index.searchsorted(t_end, side='right') 
        # Add buffer to ensure coverage for the 'next' bar logic if needed
        # TradingEnv might look ahead by 1 bar for alignment
        idx_15_end = min(idx_15_end + 5, len(df_15m)) 
        
        # 45m
        idx_45_start = df_45m.index.searchsorted(t_start)
        idx_45_end = df_45m.index.searchsorted(t_end, side='right')
        idx_45_end = min(idx_45_end + 5, len(df_45m))
        
        # Map back to array indices (assuming 1:1 mapping between df and data arrays)
        # We need to slice the arrays `data_15m`, `data_45m`
        
        try:
            episode_env = create_trading_env(
                data_5m[s:e], 
                data_15m[idx_15_start:idx_15_end], 
                data_45m[idx_45_start:idx_45_end], 
                close_prices[s:e], market_features[s:e],
                analyst_model=analyst, config=config.trading, device=device,
                market_feat_mean=mf_mean, market_feat_std=mf_std, returns=returns[s:e],
                ohlc_data=ohlc_data[s:e], timestamps=timestamps[s:e], use_analyst=True
            )
            
            # Run Backtest
            res = run_backtest(
                agent=agent, env=episode_env, 
                min_action_confidence=0.0, # Force 0.0 to ensure trades
                spread_pips=0.0, sl_atr_multiplier=config.trading.sl_atr_multiplier,
                tp_atr_multiplier=config.trading.tp_atr_multiplier,
                use_stop_loss=True, use_take_profit=True,
                min_hold_bars=0, early_exit_profit_atr=3.0, break_even_atr=2.0
            )
            
            logger.info(f"Episode {i}: {res.metrics['total_trades']} trades, Return: {res.metrics['total_return_pct']:.2f}%")

            if res.metrics['total_trades'] > 0: 
                results_list.append({
                    'return': res.metrics['total_return_pct'],
                    'drawdown': res.metrics['max_drawdown_pct'],
                    'sharpe': res.metrics['sharpe_ratio'],
                    'trades': res.metrics['total_trades'],
                    'win_rate': res.metrics['win_rate_pct'], # Corrected key
                    'expectancy': res.metrics['expectancy_pips']
                })
                # Collect Equity Curve
                if res.equity_curve is not None and len(res.equity_curve) > 0:
                    equity_curves.append(res.equity_curve)
            
            episode_env.close()
        
        except IndexError as err:
            logger.warning(f"Episode {i} Skipped: IndexError {err}")
            continue
        except Exception as e:
            logger.error(f"Error in episode {i}: {e}")
            continue

    # Analysis
    if not results_list:
        logger.error("NO EPISODES COMPLETED WITH TRADES! Check data/model.")
        sys.exit(1)

    df_res = pd.DataFrame(results_list)
    print("\n" + "="*60)
    print("MONTE CARLO RESULTS (Full Training Data)")
    print("="*60)
    print(df_res.describe())
    print("-" * 60)
    print(f"Mean Return: {df_res['return'].mean():.2f}%")
    print(f"Median Return: {df_res['return'].median():.2f}%")
    print(f"VaR (95%): {df_res['return'].quantile(0.05):.2f}%")
    print(f"Mean Max Drawdown: {df_res['drawdown'].mean():.2f}%")
    print(f"Probability of Loss: {(df_res['return'] < 0).mean()*100:.1f}%")
    
    # Save Results
    out_dir = config.paths.base_dir / 'results' / f'monte_carlo_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / 'monte_carlo_stats.csv')
    
    # Save Equity Curves Data
    np.savez_compressed(out_dir / 'equity_curves.npz', curves=np.array(equity_curves, dtype=object))

    # --- PLOTTING ---
    plt.figure(figsize=(18, 6))
    
    # 1. Distribution of Returns
    plt.subplot(1, 3, 1)
    sns.histplot(df_res['return'], kde=True, color='lime')
    plt.title('Distribution of Returns (N=50 Episodes)')
    plt.xlabel('Return (%)')
    
    # 2. Distribution of Drawdowns
    plt.subplot(1, 3, 2)
    sns.histplot(df_res['drawdown'], kde=True, color='tomato')
    plt.title('Distribution of Max Drawdowns')
    plt.xlabel('Drawdown (%)')
    
    # 3. Equity Curves (Spaghetti Plot)
    plt.subplot(1, 3, 3)
    
    # Normalize and plot all curves
    # Assume 4000 steps max, but lengths might vary if closed early (though simulation length is fixed usually)
    # We'll align by step index.
    
    max_len = max(len(c) for c in equity_curves)
    normalized_curves = []
    
    for curve in equity_curves:
        # Normalize to % Return
        norm_curve = (curve - curve[0]) / curve[0] * 100
        plt.plot(norm_curve, color='gray', alpha=0.1, linewidth=1)
        
        # Pad with NaNs for averaging if lengths differ (optional, but safe)
        if len(norm_curve) < max_len:
            padded = np.full(max_len, np.nan)
            padded[:len(norm_curve)] = norm_curve
            normalized_curves.append(padded)
        else:
            normalized_curves.append(norm_curve[:max_len])
            
    # Calculate and plot Mean Curve
    mean_curve = np.nanmean(np.array(normalized_curves), axis=0)
    plt.plot(mean_curve, color='blue', linewidth=2, label='Mean Return')
    
    plt.title('Monte Carlo Equity Trajectories')
    plt.xlabel('Bars')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'monte_carlo_dist.png')
    logger.info(f"Results saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/agent/final_model.zip')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--length', type=int, default=4000, help='Bars per episode (default 4000 ~ 2 weeks)')
    args = parser.parse_args()
    
    run_simulation(args)
