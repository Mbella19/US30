#!/usr/bin/env python3
"""Generate a monthly returns heatmap from backtest trades.csv"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure project root is on the import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config

def calculate_monthly_returns(trades_path: str, initial_balance: float = None):
    """Calculate monthly returns from trades CSV."""
    trades = pd.read_csv(trades_path)
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    trades['year'] = trades['exit_time'].dt.year
    trades['month'] = trades['exit_time'].dt.month
    
    # Calculate monthly PnL
    monthly_pnl = trades.groupby(['year', 'month'])['pnl_percent'].sum().reset_index()
    monthly_pnl.columns = ['Year', 'MonthNum', 'Return %']
    
    return monthly_pnl

def plot_heatmap(monthly_df: pd.DataFrame, save_path: str, title: str = "Monthly Returns Heatmap"):
    """Create and save the heatmap."""
    # Pivot for heatmap
    pivot = monthly_df.pivot(index='Year', columns='MonthNum', values='Return %')
    
    # Ensure all months are represented
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[sorted(pivot.columns)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Dark theme
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    # Heatmap with diverging colormap
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    vmax = min(vmax, 35)  # Cap for better color contrast
    
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
        cbar_kws={'label': 'Return %'},
        ax=ax,
        annot_kws={'fontsize': 10, 'fontweight': 'bold'}
    )
    
    # Style
    ax.set_title(title, fontsize=16, color='white', pad=20, fontweight='bold')
    ax.set_xlabel('MonthNum', fontsize=12, color='#cccccc')
    ax.set_ylabel('Year', fontsize=12, color='#cccccc')
    ax.tick_params(colors='#cccccc', which='both')
    
    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True, help="Path to trades.csv")
    parser.add_argument("--output", required=True, help="Output path for heatmap PNG")
    parser.add_argument("--title", default="Monthly Returns Heatmap (Zero Cost)", help="Chart title")
    args = parser.parse_args()
    
    monthly = calculate_monthly_returns(args.trades)
    plot_heatmap(monthly, args.output, args.title)
