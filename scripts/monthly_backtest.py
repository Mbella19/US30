#!/usr/bin/env python
"""
Monthly Backtest Analysis Script
Runs separate backtests for each calendar month and generates individual equity curves.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config
from src.models.analyst import load_analyst
from src.agents.sniper_agent import SniperAgent
from src.training.train_agent import prepare_env_data, create_trading_env
from src.evaluation.backtest import run_backtest


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def run_monthly_backtests(model_path: str = None):
    """Run backtests for each month in the dataset."""
    
    config = Config()
    device = get_device()
    
    print("=" * 60)
    print("MONTHLY BACKTEST ANALYSIS")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Spread: {config.trading.spread_pips} pips")
    print(f"Noise: {config.trading.noise_level}")
    
    # Load processed data
    print("\nLoading data...")
    df_5m = pd.read_parquet(config.paths.data_processed / 'features_5m_normalized.parquet')
    df_15m = pd.read_parquet(config.paths.data_processed / 'features_15m_normalized.parquet')
    df_45m = pd.read_parquet(config.paths.data_processed / 'features_45m_normalized.parquet')
    
    # Get feature columns (exclude OHLCV)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in df_5m.columns if c not in exclude_cols]
    
    # Load models
    analyst_path = config.paths.models_analyst / 'best.pt'
    if model_path:
        agent_path = Path(model_path)
    else:
        agent_path = config.paths.models_agent / 'final_model.zip'
    
    print(f"Agent model: {agent_path}")
    
    # Feature dims for Analyst
    feature_dims = {'5m': len(feature_cols), '15m': len(feature_cols), '45m': len(feature_cols)}
    
    # Load analyst
    use_analyst = getattr(config.trading, 'use_analyst', True)
    if use_analyst:
        analyst = load_analyst(str(analyst_path), feature_dims, device, freeze=True)
    else:
        analyst = None
    
    # Prepare full data
    lookback_5m = 48
    lookback_15m = 16
    lookback_45m = 6
    
    data_5m, data_15m, data_45m, close_prices, market_features, returns, rolling_lookback_data = prepare_env_data(
        df_5m, df_15m, df_45m, feature_cols,
        lookback_5m, lookback_15m, lookback_45m,
    )
    
    # Extract timestamps
    subsample_15m = 3
    subsample_45m = 9
    start_idx = max(
        lookback_5m,
        (lookback_15m - 1) * subsample_15m + 1,
        (lookback_45m - 1) * subsample_45m + 1,
    )
    n_samples = len(close_prices)
    timestamps_all = df_5m.index[start_idx:start_idx + n_samples]
    
    # OHLC data
    ohlc_all = None
    if all(col in df_5m.columns for col in ['open', 'high', 'low', 'close']):
        ohlc_all = df_5m[['open', 'high', 'low', 'close']].values[start_idx:start_idx + n_samples].astype(np.float32)
    
    # Compute normalization stats on first 85% (training portion)
    train_split_idx = int(0.85 * len(close_prices))
    train_market_features = market_features[:train_split_idx]
    market_feat_mean = train_market_features.mean(axis=0).astype(np.float32)
    market_feat_std = train_market_features.std(axis=0).astype(np.float32)
    market_feat_std = np.where(market_feat_std > 1e-8, market_feat_std, 1.0).astype(np.float32)
    
    # Create DataFrame for easy month-based slicing
    df_indexed = pd.DataFrame({
        'timestamp': timestamps_all,
        'idx': np.arange(n_samples)
    })
    df_indexed['year_month'] = df_indexed['timestamp'].dt.to_period('M')
    
    # Get unique months
    unique_months = df_indexed['year_month'].unique()
    print(f"\nFound {len(unique_months)} months to analyze")
    
    # Results storage
    monthly_results = []
    monthly_equity_curves = {}
    
    # Create output directory
    output_dir = config.paths.base_dir / 'results' / f'monthly_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load agent once (we'll reuse it for each month)
    from stable_baselines3.common.monitor import Monitor
    
    for i, month in enumerate(unique_months):
        print(f"\n[{i+1}/{len(unique_months)}] Processing {month}...")
        
        # Get indices for this month
        month_mask = df_indexed['year_month'] == month
        month_indices = df_indexed.loc[month_mask, 'idx'].values
        
        if len(month_indices) < 100:  # Skip months with very little data
            print(f"  Skipping - only {len(month_indices)} bars")
            continue
        
        month_start = month_indices[0]
        month_end = month_indices[-1] + 1
        
        # Slice data for this month
        month_data = (
            data_5m[month_start:month_end],
            data_15m[month_start:month_end],
            data_45m[month_start:month_end],
            close_prices[month_start:month_end],
            market_features[month_start:month_end],
        )
        
        month_returns = returns[month_start:month_end] if returns is not None else None
        month_timestamps = (timestamps_all[month_start:month_end].astype('int64') // 10**9).values
        month_ohlc = ohlc_all[month_start:month_end] if ohlc_all is not None else None
        
        # Create environment for this month
        test_config = config.trading
        test_config.noise_level = 0.0  # Ensure no noise
        
        try:
            month_env = create_trading_env(
                *month_data,
                analyst_model=analyst,
                config=test_config,
                device=device,
                market_feat_mean=market_feat_mean,
                market_feat_std=market_feat_std,
                returns=month_returns,
                ohlc_data=month_ohlc,
                timestamps=month_timestamps,
                use_analyst=use_analyst,
            )
            month_env = Monitor(month_env)
            
            # Load agent with this environment
            agent = SniperAgent.load(str(agent_path), month_env, device='cpu')
            
            # Run backtest
            results = run_backtest(
                agent=agent,
                env=month_env.unwrapped,
                min_action_confidence=config.trading.min_action_confidence,
                spread_pips=config.trading.spread_pips + config.trading.slippage_pips,
                sl_atr_multiplier=config.trading.sl_atr_multiplier,
                tp_atr_multiplier=config.trading.tp_atr_multiplier,
                use_stop_loss=config.trading.use_stop_loss,
                use_take_profit=config.trading.use_take_profit,
                min_hold_bars=config.trading.min_hold_bars,
                early_exit_profit_atr=config.trading.early_exit_profit_atr,
                break_even_atr=config.trading.break_even_atr,
            )
            
            # Store results
            metrics = results.metrics
            monthly_results.append({
                'month': str(month),
                'total_return': metrics.get('total_return_pct', 0),
                'max_drawdown': metrics.get('max_drawdown_pct', 0),
                'trades': metrics.get('total_trades', 0),
                'win_rate': metrics.get('win_rate_pct', 0),
                'sortino': metrics.get('sortino_ratio', 0),
            })
            
            # Store equity curve
            monthly_equity_curves[str(month)] = results.equity_curve
            
            print(f"  Return: {metrics.get('total_return_pct', 0):.2f}% | Trades: {metrics.get('total_trades', 0)} | DD: {metrics.get('max_drawdown_pct', 0):.2f}%")
            
            month_env.close()
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save summary CSV
    summary_df = pd.DataFrame(monthly_results)
    summary_df.to_csv(output_dir / 'monthly_summary.csv', index=False)
    print(f"\n\nSummary saved to: {output_dir / 'monthly_summary.csv'}")
    
    # Generate multi-panel equity curve plot
    print("\nGenerating equity curve plots...")
    
    n_months = len(monthly_equity_curves)
    if n_months == 0:
        print("No valid monthly backtests to plot.")
        return
    
    # Calculate grid dimensions
    cols = 4
    rows = (n_months + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.patch.set_facecolor('#0E1117')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    months_sorted = sorted(monthly_equity_curves.keys())
    
    for idx, month in enumerate(months_sorted):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        equity = monthly_equity_curves[month]
        
        ax.set_facecolor('#0E1117')
        ax.plot(equity, color='#00E5FF', linewidth=1)
        ax.axhline(y=10000, color='#666666', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.fill_between(range(len(equity)), 10000, equity, where=(equity >= 10000), color='#00E5FF', alpha=0.1)
        ax.fill_between(range(len(equity)), 10000, equity, where=(equity < 10000), color='#FF4444', alpha=0.1)
        
        # Calculate return for this month
        ret = (equity[-1] - 10000) / 100 if len(equity) > 0 else 0
        color = '#00E5FF' if ret >= 0 else '#FF4444'
        ax.set_title(f'{month}\n{ret:+.1f}%', fontsize=10, color=color, fontweight='bold')
        
        ax.tick_params(axis='both', colors='#666666', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
        ax.grid(True, color='#1E2330', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Hide unused subplots
    for idx in range(n_months, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('MONTHLY EQUITY CURVES', fontsize=16, color='white', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'monthly_equity_curves.png', dpi=150, facecolor='#0E1117', bbox_inches='tight')
    plt.close()
    
    print(f"Equity curves saved to: {output_dir / 'monthly_equity_curves.png'}")
    print(f"\n{'='*60}")
    print("MONTHLY BACKTEST COMPLETE")
    print(f"{'='*60}")
    
    # Print summary statistics
    if len(summary_df) > 0:
        winning_months = (summary_df['total_return'] > 0).sum()
        losing_months = (summary_df['total_return'] <= 0).sum()
        avg_return = summary_df['total_return'].mean()
        print(f"Winning Months: {winning_months}")
        print(f"Losing Months: {losing_months}")
        print(f"Win Rate: {winning_months / len(summary_df) * 100:.1f}%")
        print(f"Average Monthly Return: {avg_return:.2f}%")
    
    return output_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_args()
    
    run_monthly_backtests(model_path=args.model_path)
