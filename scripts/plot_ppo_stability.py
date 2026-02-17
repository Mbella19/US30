import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def moving_average(data, window_size=10):
    return data.rolling(window=window_size, min_periods=1).mean()

def plot_ppo_stability(log_dir, output_file=None):
    log_path = Path(log_dir) / "progress.csv"
    if not log_path.exists():
        print(f"Error: Could not find {log_path}")
        return

    print(f"Loading data from {log_path}...")
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check for required columns
    required_cols = [
        'time/total_timesteps', 
        'train/approx_kl', 
        'train/entropy_loss', 
        'train/clip_fraction', 
        'train/explained_variance', 
        'rollout/ep_rew_mean', 
        'train/learning_rate'
    ]
    
    # Filter for columns that actually exist
    available_cols = [c for c in required_cols if c in df.columns]
    missing_cols = set(required_cols) - set(available_cols)
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")

    if len(df) < 5:
        print("Not enough data points to plot.")
        return

    # Setup Plot
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'bmh')
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'PPO Training Stability Analysis\n{log_dir}', fontsize=16)

    steps = df['time/total_timesteps']

    # 1. KL Divergence (Policy Stability)
    ax = axes[0, 0]
    if 'train/approx_kl' in df.columns:
        kl = df['train/approx_kl']
        ax.plot(steps, kl, alpha=0.3, color='blue', label='Raw')
        ax.plot(steps, moving_average(kl, 5), color='red', linewidth=1.5, label='MA-5')
        ax.axhline(y=0.01, color='green', linestyle='--', alpha=0.7, label='Target (0.01)')
        ax.set_title('KL Divergence (Policy Stability)')
        ax.set_ylabel('KL Divergence')
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')

    # 2. Entropy (Exploration)
    # SB3 entropy_loss is usually -entropy. So we negate it to get Entropy.
    ax = axes[0, 1]
    if 'train/entropy_loss' in df.columns:
        entropy = -df['train/entropy_loss']
        ax.plot(steps, entropy, color='purple', alpha=0.4, label='Raw')
        ax.plot(steps, moving_average(entropy, 10), color='indigo', linewidth=2, label='MA-10')
        ax.set_title('Policy Entropy (Exploration → Exploitation)')
        ax.set_ylabel('Entropy')
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')

    # 3. Clip Fraction
    ax = axes[1, 0]
    if 'train/clip_fraction' in df.columns:
        clip = df['train/clip_fraction']
        ax.plot(steps, clip, color='blue', alpha=0.3)
        ax.plot(steps, moving_average(clip, 10), color='blue', linewidth=2)
        ax.set_title('PPO Clip Fraction (Updates Clipping)')
        ax.set_ylabel('Clip Fraction')
    else:
        ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')

    # 4. Explained Variance (Value Function Quality)
    ax = axes[1, 1]
    if 'train/explained_variance' in df.columns:
        ev = df['train/explained_variance']
        ax.plot(steps, ev, color='teal', alpha=0.3, label='Raw')
        ax.plot(steps, moving_average(ev, 10), color='teal', linewidth=1.5, label='MA-10')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Random (0)')
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Perfect (1)')
        ax.set_ylim(-1.0, 1.1)
        ax.set_title('Value Function Quality (Explained Variance)')
        ax.set_ylabel('EV')
        ax.legend(loc='lower right')
    else:
        ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')

    # 5. Episode Reward
    ax = axes[2, 0]
    if 'rollout/ep_rew_mean' in df.columns:
        rew = df['rollout/ep_rew_mean']
        ax.plot(steps, rew, color='green', alpha=0.6)
        ax.set_title('Mean Episode Reward')
        ax.set_ylabel('Reward')
    else:
        ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')

    # 6. Policy Gradient Loss
    ax = axes[2, 1]
    if 'train/policy_gradient_loss' in df.columns:
        pg_loss = df['train/policy_gradient_loss']
        ax.plot(steps, pg_loss, color='#F08080', alpha=0.4, label='Raw') # Light Coral
        ax.plot(steps, moving_average(pg_loss, 10), color='#8B0000', linewidth=1.5, label='MA-10') # Dark Red
        ax.set_title('Policy Gradient Loss')
        ax.set_ylabel('Loss')
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')

    # X-labels
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel('Timesteps')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if output_file is None:
        output_file = log_path.parent / "ppo_stability_analysis.png"
    
    plt.savefig(output_file, dpi=120)
    print(f"Plot saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PPO Stability from progress.csv")
    parser.add_argument("log_dir", type=str, help="Directory containing progress.csv")
    parser.add_argument("--output", type=str, default=None, help="Output image file path")
    
    args = parser.parse_args()
    plot_ppo_stability(args.log_dir, args.output)
