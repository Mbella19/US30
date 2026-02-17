
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_equity(results_dir, title):
    equity_path = os.path.join(results_dir, "equity_curve.npy")
    if not os.path.exists(equity_path):
        print(f"Error: {equity_path} not found.")
        return

    equity_curve = np.load(equity_path)
    
    plt.figure(figsize=(12, 6))
    
    # Modern dark theme
    bg_color = '#1a1a1a'
    plt.gca().set_facecolor(bg_color)
    plt.gcf().set_facecolor(bg_color)
    
    # Grid
    plt.grid(True, linestyle='--', alpha=0.2, color='white')
    
    # Plot equity
    plt.plot(equity_curve, color='#00ff88', linewidth=2, label='Equity')
    
    # Title and labels
    plt.title(title, fontsize=16, color='white', pad=20, fontweight='bold')
    plt.xlabel('Steps (15m)', fontsize=12, color='#cccccc')
    plt.ylabel('Balance ($)', fontsize=12, color='#cccccc')
    
    # Ticks
    plt.tick_params(colors='#cccccc', which='both')
    for spine in plt.gca().spines.values():
        spine.set_color('#444444')
    
    # Stats overlay
    start_bal = equity_curve[0]
    end_bal = equity_curve[-1]
    ret_pct = ((end_bal - start_bal) / start_bal) * 100
    dd = 0.0
    peak = start_bal
    for x in equity_curve:
        if x > peak: peak = x
        drawdown = (peak - x) / peak
        if drawdown > dd: dd = drawdown
    
    stats_text = f"Total Return: +{ret_pct:.1f}%\nMax Drawdown: {dd*100:.1f}%"
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes,
             color='white', fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='black', alpha=0.5, edgecolor='#333333'))
    
    # Save
    save_path = os.path.join(results_dir, "equity_curve_modern.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor=bg_color)
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Path to results directory")
    parser.add_argument("--title", required=True, help="Chart title")
    args = parser.parse_args()
    
    plot_equity(args.results_dir, args.title)
