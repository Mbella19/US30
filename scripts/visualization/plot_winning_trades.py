#!/usr/bin/env python3
"""Plot winning trades on candlestick charts."""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
from pathlib import Path

def load_ohlc_data(data_path: str) -> pd.DataFrame:
    """Load the 5-minute OHLC data."""
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    df.columns = [c.lower() for c in df.columns]
    return df

def plot_trade_candlestick(
    ohlc_df: pd.DataFrame,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    direction: str,
    pnl_pips: float,
    trade_num: int,
    output_path: str
):
    """Plot a single trade on a candlestick chart."""
    # Get data range (add some context before/after)
    start = entry_time - pd.Timedelta(hours=2)
    end = exit_time + pd.Timedelta(hours=2)
    
    trade_data = ohlc_df.loc[start:end].copy()
    if len(trade_data) < 5:
        print(f"Not enough data for trade {trade_num}")
        return
    
    # Create markers for entry and exit
    entry_marker = [np.nan] * len(trade_data)
    exit_marker = [np.nan] * len(trade_data)
    
    # Find closest indices
    entry_idx = trade_data.index.get_indexer([entry_time], method='nearest')[0]
    exit_idx = trade_data.index.get_indexer([exit_time], method='nearest')[0]
    
    entry_marker[entry_idx] = entry_price
    exit_marker[exit_idx] = exit_price
    
    # Colors
    entry_color = 'lime' if direction == 'Long' else 'red'
    exit_color = 'cyan'
    
    # Create additional plot elements
    apds = [
        mpf.make_addplot(entry_marker, type='scatter', markersize=200, 
                         marker='^' if direction == 'Long' else 'v', 
                         color=entry_color),
        mpf.make_addplot(exit_marker, type='scatter', markersize=200, 
                         marker='x', color=exit_color),
    ]
    
    # Style
    mc = mpf.make_marketcolors(
        up='#26a69a', down='#ef5350',
        edge='inherit',
        wick={'up': '#26a69a', 'down': '#ef5350'},
        volume='in'
    )
    s = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        marketcolors=mc,
        facecolor='#1a1a1a',
        figcolor='#1a1a1a',
        gridcolor='#333333',
        gridstyle='-',
        y_on_right=True
    )
    
    # Plot
    fig, axes = mpf.plot(
        trade_data,
        type='candle',
        style=s,
        title=f'Trade {trade_num}: {direction} | +{pnl_pips:.0f} pips',
        ylabel='Price',
        addplot=apds,
        figsize=(14, 8),
        returnfig=True
    )
    
    # Add legend
    axes[0].legend(['Entry', 'Exit'], loc='upper left', fontsize=10)
    
    fig.savefig(output_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True, help="Path to trades.csv")
    parser.add_argument("--ohlc", required=True, help="Path to OHLC CSV data")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    parser.add_argument("--num-trades", type=int, default=3, help="Number of trades to plot")
    parser.add_argument("--losers", action="store_true", help="Plot losing trades instead of winners")
    parser.add_argument("--breakeven", action="store_true", help="Plot breakeven trades (near zero PnL)")
    args = parser.parse_args()
    
    # Load data
    trades = pd.read_csv(args.trades)
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    
    # Filter trades
    if args.breakeven:
        filtered = trades[(trades['pnl_pips'] > -20) & (trades['pnl_pips'] < 20)].copy()
        filtered['abs_pnl'] = filtered['pnl_pips'].abs()
        filtered = filtered.sort_values('abs_pnl', ascending=True)
        trade_type = "breakeven"
    elif args.losers:
        filtered = trades[trades['pnl_pips'] < -100].sort_values('pnl_pips', ascending=True)
        trade_type = "losing"
    else:
        filtered = trades[trades['pnl_pips'] > 300].sort_values('pnl_pips', ascending=False)
        trade_type = "winning"
    
    # Take top N
    selected = filtered.head(args.num_trades)
    print(f"Selected {len(selected)} {trade_type} trades")
    
    # Load OHLC
    ohlc = load_ohlc_data(args.ohlc)
    
    # Plot each trade
    os.makedirs(args.output_dir, exist_ok=True)
    for i, (_, trade) in enumerate(selected.iterrows(), 1):
        output_path = os.path.join(args.output_dir, f"winning_trade_{i}.png")
        plot_trade_candlestick(
            ohlc,
            trade['entry_time'],
            trade['exit_time'],
            trade['entry_price'],
            trade['exit_price'],
            trade['direction'],
            trade['pnl_pips'],
            i,
            output_path
        )

if __name__ == "__main__":
    main()
