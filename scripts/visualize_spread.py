#!/usr/bin/env python3
"""
Visualize spread on a candlestick chart to see how wide it is relative to price action.

Uses settings from config/settings.py:
- InstrumentConfig for pip_value
- TradingConfig for spread_pips
"""

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path

from config.settings import config

import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Visualize spread on candlestick chart')
parser.add_argument('--spread', type=float, default=None, 
                    help='Override spread_pips value (default: use settings)')
args = parser.parse_args()


def load_data(n_candles: int = 200) -> pd.DataFrame:
    """Load recent US30 data."""
    data_path = config.paths.training_data_dir / config.data.raw_file
    print(f"Loading data from: {data_path}")
    
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Get a recent sample
    df = df.tail(n_candles).reset_index(drop=True)
    
    print(f"Loaded {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    return df


def calculate_spread_in_price_units():
    """Convert spread_pips to price units for US30."""
    # Use CLI override if provided
    spread_pips = args.spread if args.spread is not None else config.trading.spread_pips
    pip_value = config.instrument.pip_value
    
    # Spread in price units = spread_pips * pip_value
    # For US30: pip_value = 1.0 (so 1 pip = 1.0 point price move)
    spread_price = spread_pips * pip_value
    
    print(f"\n{'='*60}")
    print(f"SPREAD CONFIGURATION")
    print(f"{'='*60}")
    print(f"Instrument: {config.instrument.name}")
    print(f"Pip Value: {pip_value}")
    print(f"Spread (pips/points): {spread_pips:,.0f}")
    print(f"Spread (price units): {spread_price:,.2f} points")
    print(f"{'='*60}\n")
    
    return spread_price


def plot_candlestick_with_spread(df: pd.DataFrame, spread_price: float) -> str:
    """Create candlestick chart with spread overlay."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    facecolor='#1a1a2e')
    
    # Dark theme
    for ax in [ax1, ax2]:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#444')
    
    # --- Main Chart: Candlesticks with Spread ---
    x_indices = np.arange(len(df))
    
    # Calculate candle colors
    colors = ['#00ff88' if row['close'] >= row['open'] else '#ff4444' 
              for _, row in df.iterrows()]
    
    # Plot candlestick bodies
    width = 0.6
    for i, (_, row) in enumerate(df.iterrows()):
        open_price = row['open']
        close_price = row['close']
        high = row['high']
        low = row['low']
        color = colors[i]
        
        # Body
        body_low = min(open_price, close_price)
        body_high = max(open_price, close_price)
        body_height = max(body_high - body_low, 0.1)  # Min height for doji
        
        rect = Rectangle((i - width/2, body_low), width, body_height,
                         facecolor=color, edgecolor=color, alpha=0.9)
        ax1.add_patch(rect)
        
        # Wicks
        ax1.plot([i, i], [low, body_low], color=color, linewidth=1)
        ax1.plot([i, i], [body_high, high], color=color, linewidth=1)
    
    # Calculate price range for visualization
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_range = price_max - price_min
    
    # Add horizontal spread visualization at different price levels
    current_price = df['close'].iloc[-1]
    
    # Draw spread band (bid-ask visualization)
    # Assume current price is the mid-price
    bid_price = current_price - spread_price / 2
    ask_price = current_price + spread_price / 2
    
    # Highlight the spread zone
    ax1.axhspan(bid_price, ask_price, color='yellow', alpha=0.2, 
                label=f'Spread Zone: {spread_price:.1f} pts')
    ax1.axhline(y=current_price, color='white', linestyle='--', 
                linewidth=1, alpha=0.7, label=f'Current Price: {current_price:.2f}')
    ax1.axhline(y=bid_price, color='red', linestyle=':', 
                linewidth=1.5, alpha=0.8, label=f'Bid: {bid_price:.2f}')
    ax1.axhline(y=ask_price, color='green', linestyle=':', 
                linewidth=1.5, alpha=0.8, label=f'Ask: {ask_price:.2f}')
    
    ax1.set_xlim(-1, len(df))
    ax1.set_ylim(price_min - price_range * 0.05, price_max + price_range * 0.05)
    ax1.set_ylabel('Price (Points)', fontsize=12, color='white')
    ax1.set_title(f'{config.instrument.name} - Candlestick Chart with Spread Visualization\n'
                  f'Spread = {config.trading.spread_pips if args.spread is None else args.spread:,.0f} pips = {spread_price:.1f} points',
                  fontsize=14, color='white', fontweight='bold')
    ax1.legend(loc='upper left', facecolor='#16213e', edgecolor='#444',
               labelcolor='white', fontsize=10)
    
    # Set x-axis labels (show every N-th timestamp)
    n_labels = 10
    step = max(1, len(df) // n_labels)
    tick_positions = list(range(0, len(df), step))
    tick_labels = [df['timestamp'].iloc[i].strftime('%m-%d %H:%M') for i in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    
    # --- Bottom Chart: Candle Range vs Spread Comparison ---
    candle_ranges = df['high'] - df['low']
    
    # Bar chart showing candle ranges
    bar_colors = ['#00ff88' if r > spread_price else '#ff4444' for r in candle_ranges]
    ax2.bar(x_indices, candle_ranges, color=bar_colors, alpha=0.7, width=0.8)
    
    # Spread line
    ax2.axhline(y=spread_price, color='yellow', linestyle='-', linewidth=2, 
                label=f'Spread: {spread_price:.1f} pts')
    
    # Stats
    pct_above_spread = (candle_ranges > spread_price).mean() * 100
    avg_range = candle_ranges.mean()
    max_range = candle_ranges.max()
    
    ax2.set_xlim(-1, len(df))
    ax2.set_ylim(0, max(max_range * 1.1, spread_price * 1.5))
    ax2.set_ylabel('Candle Range (Points)', fontsize=12, color='white')
    ax2.set_xlabel('Time', fontsize=12, color='white')
    ax2.set_title(f'Candle Range vs Spread | '
                  f'Avg Range: {avg_range:.1f} pts | '
                  f'Candles > Spread: {pct_above_spread:.1f}% | '
                  f'Green = Range > Spread',
                  fontsize=11, color='white')
    ax2.legend(loc='upper right', facecolor='#16213e', edgecolor='#444',
               labelcolor='white')
    
    # Same x-axis labels
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_path = config.paths.base_dir / 'scripts' / 'spread_visualization.png'
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), 
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to: {output_path}")
    return str(output_path)


def print_spread_analysis(df: pd.DataFrame, spread_price: float):
    """Print detailed spread analysis."""
    candle_ranges = df['high'] - df['low']
    avg_range = candle_ranges.mean()
    median_range = candle_ranges.median()
    
    # Calculate ATR for comparison
    atr_period = 14
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean().iloc[-1]
    
    print(f"\n{'='*60}")
    print(f"SPREAD ANALYSIS")
    print(f"{'='*60}")
    print(f"Spread (points):          {spread_price:>10.1f}")
    print(f"Avg Candle Range (1m):    {avg_range:>10.1f}")
    print(f"Median Candle Range (1m): {median_range:>10.1f}")
    print(f"ATR ({atr_period}):                {atr:>10.1f}")
    print(f"{'='*60}")
    print(f"Spread / Avg Range:       {spread_price / avg_range:>10.1f}x")
    print(f"Spread / ATR:             {spread_price / atr:>10.1f}x")
    print(f"Candles >= Spread:        {(candle_ranges >= spread_price).mean() * 100:>9.1f}%")
    print(f"{'='*60}")
    
    if spread_price > atr * 5:
        print(f"\n⚠️  WARNING: Spread is {spread_price / atr:.1f}x ATR!")
        print(f"   This is EXTREMELY HIGH - likely set for stress testing.")
        print(f"   Typical US30 spreads: 1-5 points")
    elif spread_price > atr:
        print(f"\n⚠️  Spread is larger than ATR - may filter out small moves")
    else:
        print(f"\n✅ Spread is smaller than ATR - reasonable for trading")


def main():
    # Load data
    df = load_data(n_candles=200)
    
    # Calculate spread in price units
    spread_price = calculate_spread_in_price_units()
    
    # Print analysis
    print_spread_analysis(df, spread_price)
    
    # Create visualization
    output_path = plot_candlestick_with_spread(df, spread_price)
    
    print(f"\n✅ Visualization complete: {output_path}")


if __name__ == "__main__":
    main()
