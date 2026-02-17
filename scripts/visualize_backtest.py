import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

def plot_backtest(results_dir):
    results_path = Path(results_dir)
    equity = np.load(results_path / "equity_curve.npy")
    trades = pd.read_csv(results_path / "trades.csv")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    # Plot Equity
    ax1.plot(equity, label='Account Equity', color='blue', linewidth=1.5)
    ax1.set_title(f'Backtest Equity Curve (Final: ${equity[-1]:.2f})', fontsize=14)
    ax1.set_ylabel('Balance ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add trade markers
    # buy_trades = trades[trades['direction'] == 1]
    # sell_trades = trades[trades['direction'] == -1] # Direction is -1 for Short? Or 2? 
    # Check direction mapping. Usually 1=Long, 2=Short, 0=Flat in code, but CSV might store -1.
    # We'll skip markers for 2000 trades to avoid clutter.
    
    # Plot Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3, label='Drawdown %')
    ax2.plot(drawdown, color='red', linewidth=1)
    ax2.set_title('Drawdown (%)', fontsize=12)
    ax2.set_ylabel('Drawdown %')
    ax2.set_xlabel('Steps')
    ax2.grid(True, alpha=0.3)
    
    # Save
    output_path = results_path / "backtest_chart.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_backtest.py <results_dir>")
        sys.exit(1)
    plot_backtest(sys.argv[1])
