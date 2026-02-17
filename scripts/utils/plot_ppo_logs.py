#!/usr/bin/env python3
"""
Plot all metrics from a Stable Baselines3 PPO training log file.
"""
import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path: str) -> dict:
    """Parse SB3 PPO log file and extract all metrics."""
    metrics = {}
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Pattern to match key-value pairs in the log
    # Format: |    metric_name     | value    |
    pattern = r'\|\s+(\w+(?:/\w+)?)\s+\|\s+([-\d.e+]+)\s+\|'
    
    matches = re.findall(pattern, content)
    
    for key, value in matches:
        try:
            val = float(value)
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(val)
        except ValueError:
            continue
    
    return metrics


def plot_metrics(metrics: dict, output_dir: Path, log_name: str):
    """Create plots for all metrics."""
    
    # Group metrics by category
    categories = {
        'rollout': {},
        'train': {},
        'time': {}
    }
    
    # Mapping of non-prefixed names to their categories
    rollout_metrics = {'ep_rew_mean', 'ep_len_mean'}
    time_metrics = {'fps', 'iterations', 'time_elapsed', 'total_timesteps'}
    
    for key, values in metrics.items():
        if '/' in key:
            category, metric = key.split('/', 1)
            if category in categories:
                categories[category][metric] = values
        elif key in rollout_metrics:
            categories['rollout'][key] = values
        elif key in time_metrics:
            categories['time'][key] = values
        else:
            categories['train'][key] = values
    
    # Get timesteps for x-axis
    timesteps = metrics.get('time/total_timesteps', metrics.get('total_timesteps', None))
    if timesteps is None:
        timesteps = list(range(len(list(metrics.values())[0])))
    
    # Convert to millions for readability
    timesteps_m = [t / 1e6 for t in timesteps]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(f'PPO Training Metrics - {log_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Reward
    ax1 = fig.add_subplot(4, 2, 1)
    if 'ep_rew_mean' in categories['rollout']:
        rewards = categories['rollout']['ep_rew_mean']
        ax1.plot(timesteps_m[:len(rewards)], rewards, 'b-', alpha=0.3, label='Raw')
        # Add smoothed line
        if len(rewards) > 100:
            window = min(100, len(rewards) // 10)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(timesteps_m[window-1:len(smoothed)+window-1], smoothed, 'b-', linewidth=2, label=f'Smoothed ({window})')
        ax1.set_xlabel('Timesteps (M)')
        ax1.set_ylabel('Episode Reward Mean')
        ax1.set_title('Episode Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode Length
    ax2 = fig.add_subplot(4, 2, 2)
    if 'ep_len_mean' in categories['rollout']:
        lengths = categories['rollout']['ep_len_mean']
        ax2.plot(timesteps_m[:len(lengths)], lengths, 'g-', alpha=0.5)
        ax2.set_xlabel('Timesteps (M)')
        ax2.set_ylabel('Episode Length Mean')
        ax2.set_title('Episode Length')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Policy Gradient Loss
    ax3 = fig.add_subplot(4, 2, 3)
    if 'policy_gradient_loss' in categories['train']:
        pg_loss = categories['train']['policy_gradient_loss']
        ax3.plot(timesteps_m[:len(pg_loss)], pg_loss, 'r-', alpha=0.5)
        ax3.set_xlabel('Timesteps (M)')
        ax3.set_ylabel('Policy Gradient Loss')
        ax3.set_title('Policy Gradient Loss')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Value Loss
    ax4 = fig.add_subplot(4, 2, 4)
    if 'value_loss' in categories['train']:
        v_loss = categories['train']['value_loss']
        ax4.plot(timesteps_m[:len(v_loss)], v_loss, 'm-', alpha=0.5)
        ax4.set_xlabel('Timesteps (M)')
        ax4.set_ylabel('Value Loss')
        ax4.set_title('Value Loss')
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Entropy Loss
    ax5 = fig.add_subplot(4, 2, 5)
    if 'entropy_loss' in categories['train']:
        ent = categories['train']['entropy_loss']
        ax5.plot(timesteps_m[:len(ent)], ent, 'c-', alpha=0.5)
        ax5.set_xlabel('Timesteps (M)')
        ax5.set_ylabel('Entropy Loss')
        ax5.set_title('Entropy Loss')
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Explained Variance
    ax6 = fig.add_subplot(4, 2, 6)
    if 'explained_variance' in categories['train']:
        ev = categories['train']['explained_variance']
        ax6.plot(timesteps_m[:len(ev)], ev, 'orange', alpha=0.5)
        ax6.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect')
        ax6.axhline(y=0.0, color='r', linestyle='--', alpha=0.5, label='Random')
        ax6.set_xlabel('Timesteps (M)')
        ax6.set_ylabel('Explained Variance')
        ax6.set_title('Explained Variance')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # Plot 7: Approx KL & Clip Fraction
    ax7 = fig.add_subplot(4, 2, 7)
    if 'approx_kl' in categories['train']:
        kl = categories['train']['approx_kl']
        ax7.plot(timesteps_m[:len(kl)], kl, 'b-', alpha=0.5, label='Approx KL')
    if 'clip_fraction' in categories['train']:
        cf = categories['train']['clip_fraction']
        ax7_twin = ax7.twinx()
        ax7_twin.plot(timesteps_m[:len(cf)], cf, 'r-', alpha=0.5, label='Clip Fraction')
        ax7_twin.set_ylabel('Clip Fraction', color='r')
    ax7.set_xlabel('Timesteps (M)')
    ax7.set_ylabel('Approx KL', color='b')
    ax7.set_title('KL Divergence & Clip Fraction')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: FPS
    ax8 = fig.add_subplot(4, 2, 8)
    if 'fps' in categories['time']:
        fps = categories['time']['fps']
        ax8.plot(timesteps_m[:len(fps)], fps, 'k-', alpha=0.5)
        ax8.set_xlabel('Timesteps (M)')
        ax8.set_ylabel('FPS')
        ax8.set_title('Training Speed (FPS)')
        ax8.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    output_path = output_dir / f'{log_name}_training_metrics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    plt.show()
    
    return output_path


def print_summary(metrics: dict):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if 'time/total_timesteps' in metrics:
        total = metrics['time/total_timesteps'][-1]
        print(f"Total Timesteps: {total:,.0f} ({total/1e6:.2f}M)")
    
    if 'rollout/ep_rew_mean' in metrics:
        rewards = metrics['rollout/ep_rew_mean']
        print(f"\nEpisode Reward:")
        print(f"  Start: {rewards[0]:.2f}")
        print(f"  End:   {rewards[-1]:.2f}")
        print(f"  Max:   {max(rewards):.2f}")
        print(f"  Min:   {min(rewards):.2f}")
        
    if 'train/explained_variance' in metrics:
        ev = metrics['train/explained_variance']
        print(f"\nExplained Variance:")
        print(f"  Final: {ev[-1]:.4f}")
        print(f"  Max:   {max(ev):.4f}")
    
    if 'train/entropy_loss' in metrics:
        ent = metrics['train/entropy_loss']
        print(f"\nEntropy Loss:")
        print(f"  Start: {ent[0]:.4f}")
        print(f"  End:   {ent[-1]:.4f}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Plot PPO training logs')
    parser.add_argument('log_path', type=str, help='Path to the log.txt file')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Output directory for plots (default: same as log file)')
    args = parser.parse_args()
    
    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else log_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_name = log_path.parent.name
    
    print(f"Parsing log file: {log_path}")
    metrics = parse_log_file(log_path)
    
    print(f"Found {len(metrics)} metrics:")
    for key in sorted(metrics.keys()):
        print(f"  - {key}: {len(metrics[key])} values")
    
    print_summary(metrics)
    plot_metrics(metrics, output_dir, log_name)


if __name__ == '__main__':
    main()
