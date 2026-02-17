#!/usr/bin/env python3
"""Plot winning trades with SL and TP levels."""

import argparse
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import os
import sys

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
    entry_atr: float,
    sl_mult: float,
    tp_mult: float,
    trade_num: int,
    output_path: str
):
    """Plot a single trade on a candlestick chart."""
    # Context
    start = entry_time - pd.Timedelta(hours=4)
    end = exit_time + pd.Timedelta(hours=4)
    
    trade_data = ohlc_df.loc[start:end].copy()
    if len(trade_data) < 5:
        print(f"Not enough data for trade {trade_num}")
        return
    
    # Calculate SL/TP Levels
    if direction == 'Long':
        sl_price = entry_price - (entry_atr * sl_mult)
        tp_price = entry_price + (entry_atr * tp_mult)
    else: # Short
        sl_price = entry_price + (entry_atr * sl_mult)
        tp_price = entry_price - (entry_atr * tp_mult)

    # Markers
    entry_marker = [np.nan] * len(trade_data)
    exit_marker = [np.nan] * len(trade_data)
    
    # Indices
    try:
        entry_idx = trade_data.index.get_indexer([entry_time], method='nearest')[0]
        exit_idx = trade_data.index.get_indexer([exit_time], method='nearest')[0]
        
        entry_marker[entry_idx] = entry_price
        exit_marker[exit_idx] = exit_price
    except Exception as e:
        print(f"Error finding indices: {e}")
        return

    # SL/TP Lines (Constant horizontal lines from entry onwards?)
    # Or just lines across the whole chart? Let's do horizontal lines
    # We can use `hlines` in mpf.plot
    
    # Colors
    entry_color = 'lime' if direction == 'Long' else 'red'
    exit_color = 'cyan'
    
    # AddPlot
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
    
    # HLines for SL/TP and Entry Price
    hlines_dict = dict(hlines=[entry_price, sl_price, tp_price], 
                       colors=['white', 'red', 'green'], 
                       linestyle='-.', 
                       linewidths=[1, 1.5, 1.5],
                       alpha=0.7)

    # Plot
    fig, axes = mpf.plot(
        trade_data,
        type='candle',
        style=s,
        title=f'Trade {trade_num}: {direction} | PnL: +{pnl_pips:.0f} pips\nEntry: {entry_price:.2f} | SL: {sl_price:.2f} ({sl_mult}x) | TP: {tp_price:.2f} ({tp_mult}x)',
        ylabel='Price',
        addplot=apds,
        hlines=hlines_dict,
        figsize=(14, 8),
        returnfig=True
    )
    
    # Add legend manually via text if needed, or rely on title/colors
    # axes[0].legend(...) might loop over handles
    
    # Save
    fig.savefig(output_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True, help="Path to trades csv")
    parser.add_argument("--ohlc", required=True, help="Path to OHLC data")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument("--sl-mult", type=float, default=2.0)
    parser.add_argument("--tp-mult", type=float, default=6.0)
    parser.add_argument("--num-trades", type=int, default=5, help="Number of top winning trades to plot")
    args = parser.parse_args()
    
    # Load
    trades = pd.read_csv(args.trades)
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    
    # Filter Winners
    winners = trades[trades['pnl_pips'] > 0].copy()
    # Sort by PnL
    winners = winners.sort_values('pnl_pips', ascending=False)
    
    top_n = winners.head(args.num_trades)
    print(f"Plotting top {len(top_n)} winning trades out of {len(winners)}")
    
    ohlc = load_ohlc_data(args.ohlc)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i, (_, trade) in enumerate(top_n.iterrows(), 1):
        out_path = os.path.join(args.output_dir, f"winner_{i}_pnl_{int(trade['pnl_pips'])}.png")
        plot_trade_candlestick(
            ohlc,
            trade['entry_time'],
            trade['exit_time'],
            trade['entry_price'],
            trade['exit_price'],
            trade['direction'],
            trade['pnl_pips'],
            trade['entry_atr'],
            args.sl_mult,
            args.tp_mult,
            i,
            out_path
        )

if __name__ == "__main__":
    main()
