import re
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

def parse_training_log(log_path):
    episodes = []
    recent_rewards = []
    recent_pnls = []
    win_rates = []
    cumulative_pnls = []
    
    current_episode = None
    
    # Regex patterns
    episode_pattern = re.compile(r"Episode (\d+) Summary:")
    recent_reward_pattern = re.compile(r"Recent Avg Reward: ([\d.-]+)")
    recent_pnl_pattern = re.compile(r"Recent Avg PnL: ([\d.-]+)")
    win_rate_pattern = re.compile(r"Win Rate: ([\d.]+)%")
    
    # Pattern for cumulative PnL (usually appears in Training Progress block, but let's see if we can catch it close to episode summary)
    # The log format shows "Avg PnL: 251.32 pips" in the progress block
    cum_pnl_pattern = re.compile(r"Avg PnL: ([\d.-]+) pips")

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Iterate through lines to extract data
    # We'll use a state-machine approach or just simpler line scanning
    
    # Temporary storage for proper alignment
    temp_data = {}
    
    for i, line in enumerate(lines):
        # Check for Episode Summary Header
        ep_match = episode_pattern.search(line)
        if ep_match:
            current_episode = int(ep_match.group(1))
            temp_data = {'episode': current_episode}
            
            # Look ahead for immediate metrics in next few lines
            for j in range(1, 10): # Check next 10 lines
                if i + j >= len(lines): break
                next_line = lines[i+j]
                
                # Extract Recent Reward
                if 'reward' not in temp_data:
                    rew_match = recent_reward_pattern.search(next_line)
                    if rew_match: temp_data['reward'] = float(rew_match.group(1))
                
                # Extract Recent PnL
                if 'pnl' not in temp_data:
                    pnl_match = recent_pnl_pattern.search(next_line)
                    if pnl_match: temp_data['pnl'] = float(pnl_match.group(1))
                    
                # Extract Win Rate
                if 'win_rate' not in temp_data:
                    wr_match = win_rate_pattern.search(next_line)
                    if wr_match: temp_data['win_rate'] = float(wr_match.group(1))
            
            # Look ahead a bit further for Cumulative PnL (often in "Training Progress" block which follows Summary)
            # Or sometimes it's before?
            # In the user provided log:
            # Episode Summary ...
            # ...
            # Training Progress ...
            # ...
            # Avg PnL: ...
            
            for k in range(1, 25): # Look further ahead for training progress block
                if i + k >= len(lines): break
                future_line = lines[i+k]
                cum_match = cum_pnl_pattern.search(future_line)
                if cum_match:
                    temp_data['cum_pnl'] = float(cum_match.group(1))
                    break # Found it
            
            # Store if we found the core metrics
            if 'episode' in temp_data and 'reward' in temp_data:
                episodes.append(temp_data['episode'])
                recent_rewards.append(temp_data['reward'])
                recent_pnls.append(temp_data.get('pnl', 0.0))
                win_rates.append(temp_data.get('win_rate', 0.0))
                cumulative_pnls.append(temp_data.get('cum_pnl', np.nan)) # Use NaN if missing

    return pd.DataFrame({
        'episode': episodes,
        'recent_reward': recent_rewards,
        'recent_pnl': recent_pnls,
        'win_rate': win_rates,
        'cumulative_pnl': cumulative_pnls
    })

def plot_progress(df, output_path='training_progress.png'):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 1. PnL Plot (Recent vs Cumulative)
    ax1 = axes[0]
    ax1.plot(df['episode'], df['recent_pnl'], label='Recent Avg PnL (10 eps)', color='cyan', alpha=0.6, linewidth=1)
    ax1.plot(df['episode'], df['recent_pnl'].rolling(window=10).mean(), label='Smoothed Recent PnL', color='white', linewidth=2)
    # Plot Cumulative on secondary axis? Or just separate? Let's keep it separate or just overlay if scales match.
    # Scales might be diff. Let's start with just Recent PnL on top.
    ax1.set_title('Recent PnL Performance (Pips)', fontsize=14, color='white')
    ax1.set_ylabel('Pips')
    ax1.grid(True, alpha=0.2)
    ax1.legend()
    
    # 2. Cumulative PnL Trend
    ax2 = axes[1]
    ax2.plot(df['episode'], df['cumulative_pnl'], label='Total Avg PnL (All Time)', color='#00ff00', linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Global Average PnL Trend (Recovery Curve)', fontsize=14, color='white')
    ax2.set_ylabel('Avg PnL (Pips)')
    ax2.grid(True, alpha=0.2)
    ax2.legend()
    
    # 3. Win Rate & Reward
    ax3 = axes[2]
    # Twin axis
    ax3_right = ax3.twinx()
    
    ln1 = ax3.plot(df['episode'], df['win_rate'], label='Win Rate %', color='yellow', linewidth=1.5)
    ln2 = ax3_right.plot(df['episode'], df['recent_reward'], label='Avg Reward', color='magenta', alpha=0.7, linewidth=1.5)
    
    ax3.set_title('Win Rate & Reward Correlation', fontsize=14, color='white')
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Win Rate %')
    ax3_right.set_ylabel('Reward Value')
    
    # Combine legends
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc='center right')
    
    ax3.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # NOTE: Update log_file path below to point to your training log.
    # To find available logs: ls models/agent/training_*.log
    log_file = "models/agent/training_20251206_040548.log"
    print(f"Parsing {log_file}...")
    df = parse_training_log(log_file)
    print(f"Found {len(df)} data points.")
    
    if len(df) > 0:
        plot_progress(df, "training_progress.png")
    else:
        print("No data found to plot!")
