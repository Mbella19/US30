#!/usr/bin/env python3
"""
S/R Level Visualization on Candlestick Chart

Plots Williams Fractal-based Support/Resistance ZONES on a candlestick chart.
Uses clustering to merge nearby levels into significant zones.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict

from src.data.features import get_fractal_levels, detect_fractals, atr
from config.settings import config


def load_sample_data(n_bars: int = 500, timeframe: str = '15min') -> pd.DataFrame:
    """Load sample OHLCV data from training data."""
    data_path = config.paths.training_data_dir / config.data.raw_file
    
    print(f"Loading data from: {data_path}")
    
    # Load the CSV
    df = pd.read_csv(data_path)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Parse datetime
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
    elif 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # Ensure OHLCV columns exist
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Resample to specified timeframe (15m default for cleaner S/R)
    df = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum' if 'volume' in df.columns else 'first'
    }).dropna()
    
    # Get a sample window (most recent data)
    return df.tail(n_bars).copy()


def cluster_sr_levels(levels: np.ndarray, atr_value: float, 
                      cluster_distance_atr: float = 0.5,
                      min_touches: int = 2) -> list:
    """
    Cluster nearby S/R levels into zones.
    
    Args:
        levels: Array of price levels
        atr_value: Current ATR for distance calculation  
        cluster_distance_atr: Max distance (in ATR) to merge levels
        min_touches: Minimum touches for a zone to be significant
        
    Returns:
        List of (zone_price, touch_count) tuples, sorted by strength
    """
    if len(levels) == 0:
        return []
    
    cluster_distance = atr_value * cluster_distance_atr
    levels = np.sort(levels)
    
    zones = []
    current_cluster = [levels[0]]
    
    for level in levels[1:]:
        if level - current_cluster[-1] <= cluster_distance:
            # Add to current cluster
            current_cluster.append(level)
        else:
            # Finish current cluster, start new one
            if len(current_cluster) >= min_touches:
                zone_price = np.mean(current_cluster)
                zones.append((zone_price, len(current_cluster)))
            current_cluster = [level]
    
    # Don't forget the last cluster
    if len(current_cluster) >= min_touches:
        zone_price = np.mean(current_cluster)
        zones.append((zone_price, len(current_cluster)))
    
    # Sort by touch count (strength)
    zones.sort(key=lambda x: x[1], reverse=True)
    
    return zones


def plot_candlestick_with_sr(
    df: pd.DataFrame,
    fractal_window: int = 5,
    sr_lookback: int = 200,
    cluster_distance_atr: float = 1.0,
    min_touches: int = 2,
    max_levels: int = 4,
    save_path: str = None
):
    """
    Plot candlestick chart with clustered S/R zones.
    
    Args:
        df: OHLCV DataFrame with datetime index
        fractal_window: Williams fractal detection window
        sr_lookback: Lookback for S/R level detection
        cluster_distance_atr: Distance (in ATR) to cluster levels
        min_touches: Minimum touches for zone significance
        max_levels: Maximum S/R zones to show (per side)
        save_path: If provided, save figure to this path
    """
    # Calculate ATR for reference
    atr_values = atr(df, period=14)
    current_atr = atr_values.iloc[-1]
    
    # Get fractal-based S/R levels
    resistance_prices, support_prices = get_fractal_levels(df, fractal_window)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    
    # Convert index to numeric for plotting
    dates = df.index
    
    # Plot candlesticks
    width = 0.6
    for i in range(len(df)):
        open_price = df['open'].iloc[i]
        close_price = df['close'].iloc[i]
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]
        
        # Bullish or bearish
        if close_price >= open_price:
            color = '#00d26a'  # Green
            body_bottom = open_price
            body_height = close_price - open_price
        else:
            color = '#ff4d4d'  # Red
            body_bottom = close_price
            body_height = open_price - close_price
        
        # Wick
        ax.plot([i, i], [low_price, high_price], color=color, linewidth=0.8)
        
        # Body
        if body_height > 0:
            rect = Rectangle(
                (i - width/2, body_bottom),
                width, body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.5
            )
            ax.add_patch(rect)
        else:
            # Doji - just a line
            ax.plot([i - width/2, i + width/2], [close_price, close_price], 
                   color=color, linewidth=1.5)
    
    # Collect S/R levels from lookback window
    resistance_levels = resistance_prices.dropna().tail(sr_lookback).values
    support_levels = support_prices.dropna().tail(sr_lookback).values
    
    # Get current price for filtering
    current_price = df['close'].iloc[-1]
    
    # Cluster levels into significant zones
    resistance_zones = cluster_sr_levels(
        resistance_levels, current_atr, cluster_distance_atr, min_touches
    )
    support_zones = cluster_sr_levels(
        support_levels, current_atr, cluster_distance_atr, min_touches
    )
    
    # Filter: resistance above price, support below price
    resistance_zones = [(p, t) for p, t in resistance_zones if p > current_price * 0.995]
    support_zones = [(p, t) for p, t in support_zones if p < current_price * 1.005]
    
    # Take top N zones by strength
    resistance_zones = resistance_zones[:max_levels]
    support_zones = support_zones[:max_levels]
    
    # Zone thickness (half ATR each direction)
    zone_thickness = current_atr * 0.3
    
    # Plot resistance ZONES (semi-transparent rectangles)
    for i, (level, touches) in enumerate(resistance_zones):
        alpha = 0.3 + 0.1 * min(touches, 5)  # Stronger zones more visible
        zone_rect = Rectangle(
            (0, level - zone_thickness),
            len(df), zone_thickness * 2,
            facecolor='#ff6b6b', alpha=alpha, edgecolor='#ff6b6b',
            linewidth=1, linestyle='-'
        )
        ax.add_patch(zone_rect)
        # Label with touch count
        ax.text(len(df) + 2, level, f'R: {level:.0f} ({touches}×)', 
               color='#ff6b6b', fontsize=10, va='center', fontweight='bold')
    
    # Plot support ZONES
    for i, (level, touches) in enumerate(support_zones):
        alpha = 0.3 + 0.1 * min(touches, 5)
        zone_rect = Rectangle(
            (0, level - zone_thickness),
            len(df), zone_thickness * 2,
            facecolor='#4ecdc4', alpha=alpha, edgecolor='#4ecdc4',
            linewidth=1, linestyle='-'
        )
        ax.add_patch(zone_rect)
        ax.text(len(df) + 2, level, f'S: {level:.0f} ({touches}×)', 
               color='#4ecdc4', fontsize=10, va='center', fontweight='bold')
    
    # Styling
    ax.set_xlim(-5, len(df) + 40)
    ax.set_ylim(df['low'].min() - current_atr * 2, df['high'].max() + current_atr * 2)
    
    # X-axis with dates
    tick_positions = np.linspace(0, len(df)-1, min(10, len(df))).astype(int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([dates[i].strftime('%m/%d %H:%M') for i in tick_positions], 
                       rotation=45, ha='right', color='#8b949e')
    
    ax.tick_params(axis='y', colors='#8b949e')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    
    # Get timeframe from index
    if len(df) > 1:
        tf_minutes = (df.index[1] - df.index[0]).total_seconds() / 60
        tf_str = f"{int(tf_minutes)}m"
    else:
        tf_str = "?"
    
    # Title and labels
    ax.set_title(
        f'US30 {tf_str} - Clustered S/R Zones\n'
        f'Cluster Distance: {cluster_distance_atr}×ATR | Min Touches: {min_touches} | ATR: {current_atr:.1f}',
        fontsize=14, fontweight='bold', color='#e6edf3', pad=20
    )
    ax.set_xlabel('Time', fontsize=11, color='#8b949e')
    ax.set_ylabel('Price', fontsize=11, color='#8b949e')
    
    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor='#ff6b6b', alpha=0.5, label='Resistance Zone'),
        Patch(facecolor='#4ecdc4', alpha=0.5, label='Support Zone'),
        Patch(facecolor='#00d26a', label='Bullish Candle'),
        Patch(facecolor='#ff4d4d', label='Bearish Candle'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', 
              facecolor='#161b22', edgecolor='#30363d',
              fontsize=9, labelcolor='#e6edf3')
    
    # Add info box
    info_text = (
        f"Current Price: {current_price:.1f}\n"
        f"Resistance Zones: {len(resistance_zones)}\n"
        f"Support Zones: {len(support_zones)}"
    )
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
           fontsize=10, color='#8b949e', va='bottom', ha='right',
           bbox=dict(boxstyle='round', facecolor='#161b22', 
                    edgecolor='#30363d', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0d1117', 
                   edgecolor='none', bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    plt.show()
    
    return fig, ax


def main():
    """Main entry point."""
    print("=" * 60)
    print("S/R Zone Visualization - Clustered Williams Fractals")
    print("=" * 60)
    
    # Load data on 15-minute timeframe for cleaner S/R
    df = load_sample_data(n_bars=200, timeframe='15min')
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Clustering parameters
    fractal_window = 5      # Williams fractal detection
    sr_lookback = 200       # Look back 200 bars
    cluster_distance = 1.0  # Merge levels within 1.0 ATR
    min_touches = 2         # Zone needs 2+ touches
    max_levels = 4          # Show top 4 zones per side
    
    # Plot
    save_path = project_root / "results" / "sr_zones_chart.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_candlestick_with_sr(
        df, 
        fractal_window=fractal_window,
        sr_lookback=sr_lookback,
        cluster_distance_atr=cluster_distance,
        min_touches=min_touches,
        max_levels=max_levels,
        save_path=str(save_path)
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

