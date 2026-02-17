#!/usr/bin/env python3
"""
Run backtest on checkpoint 40M using ALL data (Training + OOS) with 0 costs.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

# Ensure project root is on the import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config, get_device
from src.agents.sniper_agent import SniperAgent
from src.evaluation.backtest import calculate_buy_and_hold, run_backtest
from src.live.bridge_constants import MODEL_FEATURE_COLS
from src.models.analyst import load_analyst
from src.training.precompute_analyst import load_cached_analyst_outputs
from src.training.train_agent import create_trading_env, prepare_env_data
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

@dataclass(frozen=True)
class BacktestDataset:
    env: Any  # Monitor-wrapped TradingEnv
    env_unwrapped: Any  # TradingEnv
    close_prices_test: np.ndarray
    timestamps_test: np.ndarray
    ohlc_test: Optional[np.ndarray]
    bh_return_pct: float
    bh_final_balance: float
    test_start_ts: pd.Timestamp
    test_end_ts: pd.Timestamp
    backtest_start: int = 0  # Index to start backtesting (after rolling warmup)

def _load_full_dataset(
    cfg: Config,
    *,
    min_action_confidence: float,
) -> BacktestDataset:
    # 1. Load Data
    processed = cfg.paths.data_processed
    df_5m = pd.read_parquet(processed / "features_5m_normalized.parquet")
    df_15m = pd.read_parquet(processed / "features_15m_normalized.parquet")
    df_45m = pd.read_parquet(processed / "features_45m_normalized.parquet")

    # 2. Ensure Feature Cols exist
    required_feature_cols = list(MODEL_FEATURE_COLS)
    for col in required_feature_cols:
        if col not in df_5m.columns: df_5m[col] = 0.0
        if col not in df_15m.columns: df_15m[col] = 0.0
        if col not in df_45m.columns: df_45m[col] = 0.0

    lookback_5m = int(cfg.analyst.lookback_5m)
    lookback_15m = int(cfg.analyst.lookback_15m)
    lookback_45m = int(cfg.analyst.lookback_45m)

    # 3. Prepare Env Data
    # This aligns the data and trims the initial lookback period
    data_5m, data_15m, data_45m, close_prices, market_features, returns, rolling_lookback_data = prepare_env_data(
        df_5m, df_15m, df_45m, required_feature_cols,
        lookback_5m, lookback_15m, lookback_45m,
    )

    # 4. Define "Test" Range = EVERYTHING (after rolling warmup)
    # prepare_env_data returns rolling_lookback_data from BEFORE start_idx, but
    # start_idx is only ~424 bars while rolling_window_size is 5760.
    # FIX: Skip first rolling_window_size bars for normalization warmup parity
    # with training (where random starts always have adequate PRIORITY 2 warmup).
    from config.settings import config as default_config
    rolling_window_size = default_config.normalization.rolling_window_size
    warmup_lookback_available = len(rolling_lookback_data) if rolling_lookback_data is not None else 0
    warmup_deficit = max(0, rolling_window_size - warmup_lookback_available)

    # The backtest starts after enough bars have accumulated for proper rolling normalization.
    # These warmup bars are passed to the env as rolling_lookback_data (from market_features).
    backtest_start = min(warmup_deficit, len(close_prices) - 1)

    # Build proper rolling_lookback_data: original + first backtest_start bars of market_features
    if backtest_start > 0:
        extra_warmup = market_features[:backtest_start]
        if rolling_lookback_data is not None and len(rolling_lookback_data) > 0:
            rolling_lookback_data = np.concatenate([rolling_lookback_data, extra_warmup], axis=0).astype(np.float32)
        else:
            rolling_lookback_data = extra_warmup.astype(np.float32)
        # Trim to rolling_window_size
        if len(rolling_lookback_data) > rolling_window_size:
            rolling_lookback_data = rolling_lookback_data[-rolling_window_size:]

    n_samples = len(close_prices)
    logger.info(
        "Rolling warmup: %d bars from lookback + %d bars skipped = %d total (target: %d)",
        warmup_lookback_available, backtest_start, len(rolling_lookback_data), rolling_window_size
    )

    # Recover timestamps
    subsample_15m = 3
    subsample_45m = 9
    start_idx = max(
        lookback_5m,
        (lookback_15m - 1) * subsample_15m + 1,
        (lookback_45m - 1) * subsample_45m + 1,
    )

    if not isinstance(df_5m.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex in processed 5m data.")

    timestamps_all = (df_5m.index[start_idx:start_idx + n_samples].astype("int64") // 10**9).values

    ohlc_all: Optional[np.ndarray] = None
    if all(c in df_5m.columns for c in ("open", "high", "low", "close")):
        ohlc_all = (
            df_5m[["open", "high", "low", "close"]]
            .values[start_idx:start_idx + n_samples]
            .astype(np.float32)
        )

    # v37 FIX: Use 70% split for normalization stats (match training pipeline)
    train_split_idx = int(0.70 * len(market_features))  # Match run_pipeline.py
    train_market = market_features[:train_split_idx]
    market_feat_mean = train_market.mean(axis=0).astype(np.float32)
    market_feat_std = train_market.std(axis=0).astype(np.float32)
    market_feat_std = np.where(market_feat_std > 1e-8, market_feat_std, 1.0).astype(np.float32)

    logger.info(
        "Normalization stats computed on first 70%% (%d samples) to match training",
        train_split_idx
    )

    # Slice - actually we take everything
    test_data = (
        data_5m,
        data_15m,
        data_45m,
        close_prices,
        market_features,
    )
    test_returns = returns
    timestamps_test = timestamps_all
    ohlc_test = ohlc_all

    # 5. Load Analyst (only if use_analyst=True)
    use_analyst = getattr(cfg.trading, 'use_analyst', True)
    analyst = None
    precomputed_cache = None
    
    if use_analyst:
        feature_dims = {"5m": len(required_feature_cols), "15m": len(required_feature_cols), "45m": len(required_feature_cols)}
        analyst_path = cfg.paths.models_analyst / "best.pt"
        analyst = load_analyst(str(analyst_path), feature_dims, device=cfg.device, freeze=True)
        logger.info("Analyst ENABLED - loading analyst model")

        # 6. Precomputed Cache
        cache_path = processed / "analyst_cache.npz"
        if cache_path.exists():
            try:
                full_cache = load_cached_analyst_outputs(str(cache_path))
                if len(full_cache["contexts"]) == len(close_prices) and len(full_cache["probs"]) == len(close_prices):
                    precomputed_cache = {
                        "contexts": full_cache["contexts"],
                        "probs": full_cache["probs"],
                    }
                    logger.info("Loaded full analyst cache.")
            except Exception as e:
                logger.warning("Failed to load analyst_cache.npz: %s", e)
    else:
        logger.info("Analyst DISABLED (use_analyst=False) - agent using market features only")

    cfg.trading.noise_level = 0.0

    env = create_trading_env(
        *test_data,
        analyst_model=analyst,
        config=cfg.trading,
        device=cfg.device,
        market_feat_mean=market_feat_mean,
        market_feat_std=market_feat_std,
        returns=test_returns,
        ohlc_data=ohlc_test,
        timestamps=timestamps_test,
        precomputed_analyst_cache=precomputed_cache,
        use_analyst=use_analyst,
        rolling_lookback_data=rolling_lookback_data,  # v37 FIX: Pass warmup data for rolling norm parity
    )

    from stable_baselines3.common.monitor import Monitor
    monitor_env = Monitor(env)

    # Buy & Hold (from backtest_start to match agent's trading window)
    bh_equity, bh_metrics = calculate_buy_and_hold(close_prices[backtest_start:], initial_balance=cfg.trading.initial_balance)
    bh_return_pct = float(bh_metrics.get("total_return_pct", 0.0))
    bh_final_balance = float(bh_metrics.get("final_balance", cfg.trading.initial_balance))

    return BacktestDataset(
        env=monitor_env,
        env_unwrapped=env,
        close_prices_test=close_prices,
        timestamps_test=timestamps_test,
        ohlc_test=ohlc_test,
        bh_return_pct=bh_return_pct,
        bh_final_balance=bh_final_balance,
        test_start_ts=pd.to_datetime(int(timestamps_test[backtest_start]), unit="s"),
        test_end_ts=pd.to_datetime(int(timestamps_test[-1]), unit="s"),
        backtest_start=backtest_start,
    )

def _resolve_checkpoint(cfg: Config, checkpoint_arg: Optional[str]) -> Path:
    if checkpoint_arg:
        checkpoint_path = Path(checkpoint_arg).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (cfg.paths.base_dir / checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    checkpoint_path = cfg.paths.models_agent / "checkpoints" / "sniper_model_40000000_steps.zip"
    if checkpoint_path.exists():
        return checkpoint_path

    found = list(cfg.paths.models_agent.rglob("*40000000*.zip"))
    if found:
        return found[0]

    raise FileNotFoundError("Could not find default 40M checkpoint in models/agents.")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a full-data backtest (train + OOS) with zero costs."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint .zip (default: 40M checkpoint in models/agents).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output filename prefix (default: checkpoint stem).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip equity curve and monthly heatmap plotting.",
    )
    return parser.parse_args()

def main():
    setup_logging(None, name=__name__)
    
    args = _parse_args()

    cfg = Config()
    cfg.device = get_device()
    
    # FORCE ZERO COSTS
    logger.info("FORCING ZERO COSTS (Spread=0, Slippage=0)")
    cfg.trading.spread_pips = 0.0
    cfg.trading.spread_pips = 0.0
    cfg.trading.slippage_pips = 0.0

    # FORCE TRIPLE RISK (3.0%)
    # Locate checkpoint
    try:
        checkpoint_path = _resolve_checkpoint(cfg, args.checkpoint)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return

    logger.info(f"Target Checkpoint: {checkpoint_path}")

    # Load Full Dataset
    # Load Full Dataset
    dataset = _load_full_dataset(cfg, min_action_confidence=0.0)

    # FORCE 3X RISK (Direct Env Modification)
    logger.info("FORCING ENV RISK parameters manually to 300.0 (3x)...")
    dataset.env_unwrapped.risk_per_trade = 300.0
    dataset.env_unwrapped.volatility_sizing = True
    logger.info(f"Env Risk Per Trade set to: {dataset.env_unwrapped.risk_per_trade}")

    
    logger.info(f"Starting Backtest on {len(dataset.timestamps_test)} bars...")
    logger.info(f"Range: {dataset.test_start_ts} -> {dataset.test_end_ts}")

    start_time = time.time()
    
    agent = SniperAgent.load(str(checkpoint_path), dataset.env, device="cpu")
    
    result = run_backtest(
        agent=agent,
        env=dataset.env_unwrapped,
        initial_balance=cfg.trading.initial_balance,
        deterministic=True,
        start_idx=dataset.backtest_start,  # Skip warmup bars for proper rolling norm
        min_action_confidence=0.0,
        spread_pips=0.0,
        sl_atr_multiplier=float(cfg.trading.sl_atr_multiplier),
        tp_atr_multiplier=float(cfg.trading.tp_atr_multiplier),
        use_stop_loss=bool(cfg.trading.use_stop_loss),
        use_take_profit=bool(cfg.trading.use_take_profit),
        min_hold_bars=int(cfg.trading.min_hold_bars),
        break_even_atr=float(cfg.trading.break_even_atr),
        early_exit_profit_atr=float(cfg.trading.early_exit_profit_atr),
    )
    
    elapsed = time.time() - start_time
    metrics = result.metrics
    
    print("\n" + "="*60)
    print(f"RESULTS FOR Checkpoint {checkpoint_path.name}")
    print(f"Dataset: FULL (Training + OOS), skipped {dataset.backtest_start} warmup bars")
    print(f"Costs: 0.0 spread, 0.0 slippage")
    print("="*60)
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Net PnL: ${metrics['net_pnl_cash']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino: {metrics['sortino_ratio']:.2f}")
    print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print("-" * 60)
    print(f"Buy & Hold Return: {dataset.bh_return_pct:.2f}%")
    print(f"Beats B&H: {metrics['total_return_pct'] > dataset.bh_return_pct}")
    print("="*60 + "\n")

    # --- PLOTTING ---
    if args.no_plots:
        return

    output_prefix = args.output_prefix or checkpoint_path.stem

    print("\n[INFO] Starting Plotting phase...")
    try:
        print("[INFO] Importing matplotlib...")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Determine OOS Split Timestamp
        total_len = len(dataset.timestamps_test)
        split_idx = int(0.85 * total_len)
        print(f"[INFO] Total len: {total_len}, Split idx: {split_idx}")
        
        # OOS Start Time
        oos_timestamp = dataset.timestamps_test[split_idx]
        oos_dt = pd.to_datetime(oos_timestamp, unit='s')
        print(f"[INFO] OOS Date: {oos_dt}")
        
        dates = [pd.to_datetime(ts, unit='s') for ts in result.timestamps]
        equity = result.equity_curve
        print(f"[INFO] Dates len: {len(dates)}, Equity len: {len(equity)}")
        
        # Basic Validation
        if len(dates) != len(equity):
            logger.warning(f"Plotting mismatch: dates={len(dates)}, equity={len(equity)}. Trimming to min.")
            n = min(len(dates), len(equity))
            dates = dates[:n]
            equity = equity[:n]

        # Calculate Buy & Hold Curve
        start_price = dataset.close_prices_test[0]
        start_bal = equity[0] # Should be 10000
        bh_curve = (start_bal / start_price) * dataset.close_prices_test

        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity, label='Agent Equity', linewidth=1.5, color='blue')
        
        if len(dataset.timestamps_test) == len(bh_curve):
             bh_dates = [pd.to_datetime(ts, unit='s') for ts in dataset.timestamps_test]
             plt.plot(bh_dates, bh_curve, label='Buy & Hold', linewidth=1.0, color='gray', linestyle='--', alpha=0.7)

        # OOS Line
        plt.axvline(x=oos_dt, color='red', linestyle='--', linewidth=2, label='OOS Start (85%)')
        
        plt.title(
            f"Equity Curve - {checkpoint_path.name} (Full Data, 0 Cost)\n"
            f"Total Return: {metrics['total_return_pct']:.2f}% vs B&H: {dataset.bh_return_pct:.2f}%"
        )
        plt.xlabel('Date')
        plt.ylabel('Balance ($)')
        plt.legend(loc='upper left')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        output_plot = cfg.paths.base_dir / "results" / f"equity_curve_{output_prefix}_full_0cost.png"
        print(f"[INFO] Saving plot to: {output_plot}")
        plt.savefig(output_plot)
        print("[INFO] Plot saved successfully.")

        
        # --- SAVE TRADES TO CSV ---
        print("\n[INFO] Saving trades to CSV...")
        all_trades_data = []
        for t in result.trades:
            all_trades_data.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'direction': 'Long' if t.direction == 1 else 'Short',
                'size': t.size,
                'pnl_pips': t.pnl_pips,
                'pnl_percent': t.pnl_percent,
                'entry_atr': t.entry_atr
            })
        
        trades_df_out = pd.DataFrame(all_trades_data)
        trades_csv_path = cfg.paths.base_dir / "results" / f"trades_{output_prefix}.csv"
        trades_df_out.to_csv(trades_csv_path, index=False)
        print(f"[INFO] Saved trades to: {trades_csv_path}")

        # --- MONTHLY HEATMAP ---
        print("\n[INFO] Generating Monthly Heatmap...")
        import seaborn as sns
        
        # Calculate Monthly Returns based on FIXED starting balance per month
        # This ignores compounding.
        # Formula: Sum(Trade PnL $) / initial_balance * 100.0
        
        # 1. Calculate Dollar PnL for each trade
        # PnL($) = pips * pip_value * lot_size * point_multiplier
        pip_val = cfg.instrument.pip_value
        lot_sz = cfg.instrument.lot_size
        pt_mult = cfg.instrument.point_multiplier
        
        trades_data = []
        for t in result.trades:
            pnl_dollars = t.pnl_pips * pip_val * lot_sz * pt_mult
            trades_data.append({
                'exit_time': t.exit_time,
                'pnl_dollars': pnl_dollars
            })
            
        if not trades_data:
            print("[WARN] No trades found for heatmap.")
        else:
            trades_df = pd.DataFrame(trades_data)
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            trades_df['year'] = trades_df['exit_time'].dt.year
            trades_df['month'] = trades_df['exit_time'].dt.month
            
            # Group by Year/Month and SUM the Dollar PnL
            monthly_pnl = trades_df.groupby(['year', 'month'])['pnl_dollars'].sum().reset_index()
            
            # Calculate % Return relative to initial balance
            monthly_pnl['return_pct'] = (monthly_pnl['pnl_dollars'] / cfg.trading.initial_balance) * 100.0
            
            # Pivot
            pivot = monthly_pnl.pivot(index='year', columns='month', values='return_pct')
            
            # Fill missing months
            for m in range(1, 13):
                if m not in pivot.columns:
                    pivot[m] = np.nan
            pivot = pivot[sorted(pivot.columns)]
            
            # Plot
            fig, ax = plt.subplots(figsize=(14, 8))
            fig.patch.set_facecolor('#1a1a1a')
            ax.set_facecolor('#1a1a1a')
            
            vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
            vmax = min(vmax, 35) # Cap at 35%
            
            sns.heatmap(
                pivot, 
                annot=True, 
                fmt='.1f', 
                cmap='RdYlGn', 
                center=0,
                vmin=-vmax, 
                vmax=vmax,
                linewidths=0.5, 
                linecolor='#333333',
                cbar_kws={'label': 'Return % (Basis $10k)'},
                ax=ax,
                annot_kws={'fontsize': 10, 'fontweight': 'bold'}
            )
            
            ax.set_title(
                f"Monthly Returns Heatmap ({checkpoint_path.name} - Non-Compounded)",
                fontsize=16,
                color='white',
                pad=20,
                fontweight='bold',
            )
        ax.set_xlabel('Month', fontsize=12, color='#cccccc')
        ax.set_ylabel('Year', fontsize=12, color='#cccccc')
        ax.tick_params(colors='#cccccc', which='both')
        
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        
        plt.tight_layout()
        output_heatmap = cfg.paths.base_dir / "results" / f"monthly_heatmap_{output_prefix}_0cost.png"
        print(f"[INFO] Saving heatmap to: {output_heatmap}")
        plt.savefig(output_heatmap, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print("[INFO] Heatmap saved successfully.")
    
    except Exception as e:
        print(f"[ERROR] Plotting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
