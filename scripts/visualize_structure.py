
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add src to path
sys.path.append(str(Path.cwd()))

from src.data.features import detect_fractals, detect_structure_breaks

def visualize_structure():
    print("Loading data...")
    df = pd.read_parquet('data/processed/features_15m.parquet')
    
    # Select a slice with some action
    # Let's try to find a place with decent volatility
    start_idx = 1000
    end_idx = 1200
    df_slice = df.iloc[start_idx:end_idx].copy()
    
    print("Detecting features...")
    f_high, f_low = detect_fractals(df_slice, n=5)
    struct_df = detect_structure_breaks(df_slice, f_high, f_low, n=5)
    
    # Combine
    df_slice['fractal_high'] = f_high
    df_slice['fractal_low'] = f_low
    for col in struct_df.columns:
        df_slice[col] = struct_df[col]
        
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot Candles (Simplified)
    width = 0.005 # Width of candle bars in days (approx 7 mins for 15m bars?)
    # Actually, let's just use index for x-axis to avoid gaps
    x = np.arange(len(df_slice))
    
    # Up candles
    up = df_slice[df_slice.close >= df_slice.open]
    down = df_slice[df_slice.close < df_slice.open]
    
    # Plot wicks
    ax.vlines(x, df_slice.low, df_slice.high, color='gray', linewidth=1)
    
    # Plot bodies
    # We need to map index to x array
    up_idx = [df_slice.index.get_loc(idx) for idx in up.index]
    down_idx = [df_slice.index.get_loc(idx) for idx in down.index]
    
    # Correction: df_slice is a slice, so get_loc might be tricky if we don't reset index or use integer indexing
    # Let's just use integer indexing from 0 to len
    up_indices = np.where(df_slice.close >= df_slice.open)[0]
    down_indices = np.where(df_slice.close < df_slice.open)[0]
    
    ax.bar(up_indices, up.close - up.open, bottom=up.open, color='green', width=0.6)
    ax.bar(down_indices, down.close - down.open, bottom=down.open, color='red', width=0.6)
    
    # Plot Fractals
    fractal_high_indices = np.where(df_slice.fractal_high)[0]
    fractal_low_indices = np.where(df_slice.fractal_low)[0]
    
    # Note: Fractals are marked at the CONFIRMATION candle (i.e. 2 bars later)
    # But the actual fractal is at i - 2.
    # Let's verify where detect_fractals marks them.
    # "We mark at CURRENT position (i), indicating we NOW KNOW about this fractal"
    # "The actual S/R level is at df['high'].iloc[fractal_idx]" where fractal_idx = i - half_n
    
    # So if we want to plot the marker ON the fractal candle, we need to shift back by 2
    # But for "System Awareness", the marker is correct at the confirmation time.
    # Let's plot the marker at the ACTUAL fractal high/low for visual clarity, 
    # but maybe add a small dot at confirmation time?
    # Let's stick to plotting on the fractal candle itself for "is the logic ok" check.
    
    # Shift indices back by 2 (n//2)
    offset = 2
    
    # Filter out indices that would be negative
    valid_high_indices = fractal_high_indices[fractal_high_indices >= offset]
    valid_low_indices = fractal_low_indices[fractal_low_indices >= offset]
    
    ax.scatter(valid_high_indices - offset, df_slice.high.iloc[valid_high_indices - offset] + 0.0002, 
               marker='v', color='green', s=100, label='Fractal High')
    ax.scatter(valid_low_indices - offset, df_slice.low.iloc[valid_low_indices - offset] - 0.0002, 
               marker='^', color='red', s=100, label='Fractal Low')
               
    # Plot continuous structure features (v2)
    # structure_fade: positive = bullish, negative = bearish
    if 'structure_fade' in df_slice.columns:
        ax2 = ax.twinx()
        ax2.plot(x, df_slice.structure_fade.values, color='purple', alpha=0.5, linewidth=1.5, label='structure_fade')
        ax2.set_ylabel('Structure Fade', color='purple')
        ax2.axhline(y=0, color='purple', linestyle=':', alpha=0.3)
        ax2.legend(loc='upper right')

    # Plot BOS/CHoCH magnitude as vertical bars where they occur
    if 'bos_magnitude' in df_slice.columns:
        bos_active = df_slice.bos_magnitude.values != 0
        bos_idx = np.where(bos_active)[0]
        for idx in bos_idx:
            mag = df_slice.bos_magnitude.iloc[idx]
            color = 'green' if mag > 0 else 'red'
            ax.axvline(x=idx, color=color, linestyle='--', alpha=0.4)
            ax.text(idx, df_slice.high.iloc[idx] + 0.0005, f'BOS\n{mag:.2f}', color=color, rotation=90, fontsize=7)

    if 'choch_magnitude' in df_slice.columns:
        choch_active = df_slice.choch_magnitude.values != 0
        choch_idx = np.where(choch_active)[0]
        for idx in choch_idx:
            mag = df_slice.choch_magnitude.iloc[idx]
            color = 'blue' if mag > 0 else 'orange'
            ax.axvline(x=idx, color=color, linestyle='-', alpha=0.5)
            ax.text(idx, df_slice.low.iloc[idx] - 0.0005, f'CHoCH\n{mag:.2f}', color=color, rotation=90, fontsize=7)

    plt.title('Market Structure Analysis (Fractals, Structure Fade, BOS/CHoCH Magnitude)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = 'structure_visualization.png'
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    visualize_structure()
