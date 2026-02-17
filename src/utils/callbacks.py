"""
Shared callback utilities for RL training.

This module consolidates commonly used callbacks to avoid code duplication
across training scripts.
"""

import gc
from typing import Callable, Optional, Dict, List
from pathlib import Path
from datetime import datetime
import json

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback

from .logging_config import get_logger

logger = get_logger(__name__)


def linear_schedule(initial_value: float, final_value: float = None) -> Callable[[float], float]:
    """
    Linear learning rate schedule for PPO.

    Args:
        initial_value: Starting learning rate
        final_value: Ending learning rate (default: 5% of initial)

    Returns:
        A function that takes progress (1.0 -> 0.0) and returns current LR
    """
    if final_value is None:
        final_value = initial_value * 0.05  # Decay to 5% by default

    def schedule(progress_remaining: float) -> float:
        """
        progress_remaining goes from 1.0 (start) to 0.0 (end)
        """
        return final_value + progress_remaining * (initial_value - final_value)

    return schedule


class MemoryCleanupCallback(BaseCallback):
    """
    Callback to periodically clean up memory during training.

    Essential for Apple M2 with limited 8GB RAM.
    """

    def __init__(self, cleanup_freq: int = 20000, verbose: int = 0):
        """
        Initialize memory cleanup callback.

        Args:
            cleanup_freq: Steps between memory cleanups. Default 20000 (increased from
                          5000 for better throughput with vectorized environments).
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.cleanup_freq = cleanup_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.cleanup_freq == 0:
            gc.collect()
            device_type = getattr(getattr(self.model, "device", None), "type", None)
            if device_type == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.verbose > 0:
                print(f"Memory cleanup at step {self.n_calls}")
        return True


class AgentTrainingLogger(BaseCallback):
    """
    Custom callback for detailed agent training logging.

    Tracks:
    - Episode rewards
    - Action distributions
    - PnL statistics
    - Win rate evolution
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_freq: int = 1000,
        checkpoint_plot_freq: int = 500_000,
        reward_plot_downsample: int = 1_000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_freq = log_freq
        self.checkpoint_plot_freq = checkpoint_plot_freq
        self.reward_plot_downsample = max(1, int(reward_plot_downsample))

        # Tracking variables
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_pnls: List[float] = []
        self.episode_trades: List[int] = []
        self.episode_win_rates: List[float] = []
        self.episode_avg_trade_durations: List[float] = []
        self.action_counts = {0: 0, 1: 0, 2: 0}  # Flat, Long, Short
        self.size_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Position sizes
        self._cumulative_reward: float = 0.0
        self._cumulative_reward_timesteps: List[int] = []
        self._cumulative_reward_values: List[float] = []
        self._last_reward_record_timestep: int = 0

        # Current episode tracking
        self.current_ep_reward = 0
        self.current_ep_length = 0
        self.current_ep_actions = []

        # Training start time
        self.start_time = None

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _on_training_start(self):
        self.start_time = datetime.now()
        self._maybe_load_cumulative_reward_history()

        if not self._cumulative_reward_timesteps:
            start_t = int(getattr(self, "num_timesteps", 0))
            self._cumulative_reward_timesteps.append(start_t)
            self._cumulative_reward_values.append(float(self._cumulative_reward))
            self._last_reward_record_timestep = start_t

        logger.info("=" * 70)
        logger.info("PPO AGENT TRAINING STARTED")
        logger.info("=" * 70)
        logger.info(f"Total timesteps: {self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else 'N/A'}")
        logger.info(f"Log directory: {self.log_dir}")
        logger.info("=" * 70)

    def _on_step(self) -> bool:
        # Track rewards
        if len(self.locals.get('rewards', [])) > 0:
            reward = float(np.asarray(self.locals['rewards'])[0])
            self.current_ep_reward += reward
            self.current_ep_length += 1
            self._cumulative_reward += reward

            timestep = int(getattr(self, "num_timesteps", self.n_calls))
            if (timestep - self._last_reward_record_timestep) >= self.reward_plot_downsample:
                self._cumulative_reward_timesteps.append(timestep)
                self._cumulative_reward_values.append(float(self._cumulative_reward))
                self._last_reward_record_timestep = timestep

        # Plot cumulative reward after each checkpoint save
        if (
            self.log_dir
            and self.checkpoint_plot_freq > 0
            and (self.n_calls % self.checkpoint_plot_freq == 0)
        ):
            timestep = int(getattr(self, "num_timesteps", self.n_calls))
            self._save_cumulative_reward_plot(timestep)
            self._save_cumulative_reward_history()

        # Track actions (direction and size for MultiDiscrete)
        if len(self.locals.get('actions', [])) > 0:
            action = self.locals['actions'][0]
            if isinstance(action, np.ndarray):
                if len(action) >= 2:
                    direction = int(action[0])
                    size = int(action[1])
                    self.action_counts[direction] = self.action_counts.get(direction, 0) + 1
                    self.size_counts[size] = self.size_counts.get(size, 0) + 1
                    self.current_ep_actions.append(direction)
                else:
                    direction = int(action[0]) if len(action) == 1 else int(action)
                    self.action_counts[direction] = self.action_counts.get(direction, 0) + 1
                    self.current_ep_actions.append(direction)
            elif isinstance(action, (int, np.integer)):
                self.action_counts[int(action)] = self.action_counts.get(int(action), 0) + 1
                self.current_ep_actions.append(int(action))

        # Check for episode done
        dones = self.locals.get('dones', [False])
        if any(dones):
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)

            infos = self.locals.get('infos', [{}])
            if len(infos) > 0 and infos[0]:
                info = infos[0]
                self.episode_pnls.append(info.get('total_pnl', 0.0))
                self.episode_trades.append(info.get('n_trades', 0))
                n_trades = info.get('n_trades', 0)
                win_rate = 0.0
                avg_duration = 0.0
                if n_trades > 0 and 'trades' in info:
                    wins = sum(1 for t in info['trades'] if t.get('pnl', 0) > 0)
                    win_rate = wins / n_trades
                    durations = [t.get('bars_held', 0) for t in info['trades'] if 'bars_held' in t]
                    if durations:
                        avg_duration = sum(durations) / len(durations)
                self.episode_win_rates.append(win_rate)
                self.episode_avg_trade_durations.append(avg_duration)

            n_episodes = len(self.episode_rewards)
            if n_episodes % 10 == 0:
                self._log_episode_summary(n_episodes)
                self._update_live_equity_curve()

            self.current_ep_reward = 0
            self.current_ep_length = 0
            self.current_ep_actions = []

        if self.n_calls % self.log_freq == 0:
            self._log_training_progress()

        return True

    def _log_episode_summary(self, n_episodes: int):
        """Log summary for recent episodes."""
        recent_rewards = self.episode_rewards[-10:]
        recent_pnls = self.episode_pnls[-10:] if self.episode_pnls else [0]

        logger.info("-" * 50)
        logger.info(f"Episode {n_episodes} Summary:")
        logger.info(f"  Recent Avg Reward: {np.mean(recent_rewards):.2f}")
        logger.info(f"  Recent Avg PnL: {np.mean(recent_pnls):.2f} pips")
        if self.episode_trades:
            recent_trades = self.episode_trades[-10:]
            logger.info(f"  Recent Avg Trades: {np.mean(recent_trades):.1f}")

        if self.episode_win_rates:
            logger.info(f"  Win Rate: {self.episode_win_rates[-1]*100:.1f}%")

        if self.episode_avg_trade_durations:
            logger.info(f"  Avg Trade Duration: {self.episode_avg_trade_durations[-1]:.1f} bars")

    def _log_training_progress(self):
        """Log overall training progress."""
        total_actions = sum(self.action_counts.values())
        if total_actions == 0:
            return

        action_pcts = {k: v/total_actions*100 for k, v in self.action_counts.items()}

        logger.info("-" * 50)
        logger.info(f"Training Progress @ {self.n_calls} steps:")
        logger.info(f"  Episodes completed: {len(self.episode_rewards)}")
        logger.info(f"  Action Distribution: Flat={action_pcts.get(0, 0):.1f}%, "
                   f"Long={action_pcts.get(1, 0):.1f}%, Short={action_pcts.get(2, 0):.1f}%")

        if self.episode_rewards:
            logger.info(f"  Avg Episode Reward: {np.mean(self.episode_rewards):.2f}")
            logger.info(f"  Max Episode Reward: {np.max(self.episode_rewards):.2f}")
            logger.info(f"  Min Episode Reward: {np.min(self.episode_rewards):.2f}")

        if self.episode_pnls:
            logger.info(f"  Avg PnL: {np.mean(self.episode_pnls):.2f} pips")

        # Memory cleanup
        device_type = getattr(getattr(self.model, 'device', None), 'type', None)
        if device_type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _update_live_equity_curve(self):
        """Update live equity curve plot during training."""
        if not self.log_dir or len(self.episode_pnls) < 10:
            return

        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            cumulative_pnl = np.cumsum(self.episode_pnls)
            ax1.plot(cumulative_pnl, 'b-', linewidth=1.5, alpha=0.8)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl,
                            where=np.array(cumulative_pnl) > 0, alpha=0.3, color='green')
            ax1.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl,
                            where=np.array(cumulative_pnl) < 0, alpha=0.3, color='red')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Cumulative PnL (pips)')
            ax1.set_title(f'Live Training Equity Curve - {len(self.episode_pnls)} episodes')
            ax1.grid(True, alpha=0.3)

            total_actions = sum(self.action_counts.values())
            if total_actions > 0:
                labels = ['Flat', 'Long', 'Short']
                sizes = [self.action_counts.get(i, 0) for i in range(3)]
                colors = ['gray', 'green', 'red']
                ax2.bar(labels, sizes, color=colors, alpha=0.7)
                for i, v in enumerate(sizes):
                    pct = v / total_actions * 100
                    ax2.text(i, v + max(sizes)*0.02, f'{pct:.1f}%', ha='center', fontsize=10)
                ax2.set_ylabel('Action Count')
                ax2.set_title('Action Distribution')

            plt.tight_layout()
            plt.savefig(self.log_dir / 'live_equity_curve.png', dpi=100, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            logger.debug(f"Failed to update live equity curve: {e}")

    def _on_training_end(self):
        """Log final training summary and create visualizations."""
        total_time = (datetime.now() - self.start_time).total_seconds()

        if self.log_dir:
            timestep = int(getattr(self, "num_timesteps", self.n_calls))
            self._save_cumulative_reward_plot(timestep, is_final=True)
            self._save_cumulative_reward_history()

        logger.info("=" * 70)
        logger.info("PPO AGENT TRAINING COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Total Training Time: {total_time/60:.1f} minutes")
        logger.info(f"Total Episodes: {len(self.episode_rewards)}")
        logger.info(f"Total Timesteps: {self.n_calls}")

        if self.episode_rewards:
            logger.info(f"Final Avg Reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")

        if self.episode_pnls:
            logger.info(f"Final Avg PnL (last 100): {np.mean(self.episode_pnls[-100:]):.2f} pips")

        logger.info("=" * 70)

        if self.log_dir:
            self._save_metrics()
        self._create_visualizations()

    def _cumulative_reward_history_path(self) -> Optional[Path]:
        if not self.log_dir:
            return None
        return self.log_dir / "cumulative_reward_history.npz"

    def _maybe_load_cumulative_reward_history(self) -> None:
        """Load cumulative reward history if present (useful when resuming training)."""
        path = self._cumulative_reward_history_path()
        if path is None or not path.is_file():
            return
        try:
            with np.load(path) as data:
                if "timesteps" not in data or "cumulative_reward" not in data:
                    return
                timesteps = data["timesteps"]
                values = data["cumulative_reward"]

            self._cumulative_reward_timesteps = [int(t) for t in np.asarray(timesteps).tolist()]
            self._cumulative_reward_values = [float(v) for v in np.asarray(values).tolist()]
            if self._cumulative_reward_values and self._cumulative_reward_timesteps:
                self._cumulative_reward = float(self._cumulative_reward_values[-1])
                self._last_reward_record_timestep = int(self._cumulative_reward_timesteps[-1])
        except Exception as exc:
            logger.warning(f"Failed to load cumulative reward history: {exc}")

    def _save_cumulative_reward_history(self) -> None:
        path = self._cumulative_reward_history_path()
        if path is None:
            return
        try:
            np.savez_compressed(
                path,
                timesteps=np.asarray(self._cumulative_reward_timesteps, dtype=np.int64),
                cumulative_reward=np.asarray(self._cumulative_reward_values, dtype=np.float32),
            )
        except Exception as exc:
            logger.warning(f"Failed to save cumulative reward history: {exc}")

    def _save_cumulative_reward_plot(self, timestep: int, is_final: bool = False) -> None:
        if not self.log_dir:
            return
        if len(self._cumulative_reward_timesteps) < 2:
            return

        checkpoint_dir = self.log_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        latest_path = checkpoint_dir / "cumulative_reward_latest.png"

        try:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(self._cumulative_reward_timesteps, self._cumulative_reward_values, color="dodgerblue", linewidth=1.5)
            ax.set_title(f"Cumulative Reward vs Timesteps (Step {timestep:,})")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Cumulative Reward")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            fig.savefig(latest_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            logger.warning(f"Failed to save cumulative reward plot: {exc}")

    def _save_metrics(self):
        """Save training metrics to JSON."""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_pnls': self.episode_pnls,
            'episode_trades': self.episode_trades,
            'episode_win_rates': self.episode_win_rates,
            'action_counts': self.action_counts,
            'total_timesteps': self.n_calls,
            'total_episodes': len(self.episode_rewards)
        }

        metrics_path = self.log_dir / 'agent_training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to: {metrics_path}")

    def _create_visualizations(self):
        """Create training visualizations."""
        if len(self.episode_rewards) < 10:
            logger.warning("Not enough episodes for visualizations")
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Episode rewards
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, color='blue')
        window = min(50, len(self.episode_rewards) // 5 + 1)
        smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(self.episode_rewards)), smoothed,
               color='red', linewidth=2, label=f'Smoothed (w={window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # PnL per episode
        ax = axes[0, 1]
        if self.episode_pnls:
            ax.plot(self.episode_pnls, alpha=0.3, color='green')
            if len(self.episode_pnls) >= window:
                smoothed_pnl = np.convolve(self.episode_pnls, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.episode_pnls)), smoothed_pnl,
                       color='darkgreen', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('PnL (pips)')
        ax.set_title('Episode PnL')
        ax.grid(True, alpha=0.3)

        # Win rate evolution
        ax = axes[0, 2]
        if self.episode_win_rates:
            ax.plot(self.episode_win_rates, alpha=0.5, color='purple')
            ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='50%')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate Evolution')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Action distribution
        ax = axes[1, 0]
        action_names = ['Flat', 'Long', 'Short']
        action_vals = [self.action_counts.get(i, 0) for i in range(3)]
        colors = ['gray', 'green', 'red']
        bars = ax.bar(action_names, action_vals, color=colors)
        ax.set_ylabel('Count')
        ax.set_title('Action Distribution')
        total = sum(action_vals)
        if total > 0:
            for bar, val in zip(bars, action_vals):
                pct = val / total * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{pct:.1f}%', ha='center', va='bottom')

        # Episode length distribution
        ax = axes[1, 1]
        ax.hist(self.episode_lengths, bins=30, color='blue', alpha=0.7)
        ax.axvline(x=np.mean(self.episode_lengths), color='red',
                  linestyle='--', label=f'Mean: {np.mean(self.episode_lengths):.0f}')
        ax.set_xlabel('Episode Length')
        ax.set_ylabel('Count')
        ax.set_title('Episode Length Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cumulative PnL
        ax = axes[1, 2]
        if self.episode_pnls:
            cumulative_pnl = np.cumsum(self.episode_pnls)
            ax.plot(cumulative_pnl, color='green', linewidth=2)
            ax.fill_between(range(len(cumulative_pnl)), cumulative_pnl, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative PnL (pips)')
        ax.set_title('Cumulative PnL')
        ax.grid(True, alpha=0.3)

        plt.suptitle('PPO Agent Training Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = self.log_dir / 'agent_training_summary.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Training visualization saved to: {save_path}")

    def get_metrics(self) -> Dict:
        """Get all tracked metrics."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_pnls': self.episode_pnls,
            'episode_trades': self.episode_trades,
            'episode_win_rates': self.episode_win_rates,
            'action_counts': self.action_counts
        }


class GradientNormCallback(BaseCallback):
    """
    Level 2 gradient monitoring: tracks gradient norm per layer during training.

    Registers backward hooks on every trainable parameter so that gradient norms
    are captured on every backward pass (every minibatch of every PPO epoch).
    Periodically logs mean norms to TensorBoard + console, and saves PNG plots.

    Answers: "Are gradients flowing to all layers? Is the critic learning faster
    than the actor? Are any layers stuck?"

    Zero overhead between log intervals — hooks just accumulate a single float.
    """

    # Friendly name mapping for SB3 PPO internal parameter paths
    _NAME_MAP = {
        'features_extractor': 'input',
        'mlp_extractor.policy_net': 'actor',
        'mlp_extractor.value_net': 'critic',
        'action_net': 'actor_head',
        'value_net': 'critic_head',
    }

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_freq: int = 50_000,
        verbose: int = 0,
    ):
        """
        Args:
            log_dir: Directory for PNG output.
            log_freq: Log gradient norms every N timesteps.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_freq = log_freq

        # Accumulated gradient norms between log intervals
        # layer_name -> list of L2 norms (one per backward pass)
        self._grad_norms: Dict[str, List[float]] = {}
        self._hooks: list = []

        # History for evolution plots: list of {timestep, norms_dict}
        self.history: List[Dict] = []

    def _friendly_name(self, param_name: str) -> str:
        """Convert SB3 internal param name to readable layer name."""
        for prefix, friendly in self._NAME_MAP.items():
            if param_name.startswith(prefix):
                rest = param_name[len(prefix):]
                # e.g. ".0.weight" -> "/layer_0/weight"
                rest = rest.replace('.', '/')
                return f"{friendly}{rest}"
        return param_name.replace('.', '/')

    def _on_training_start(self) -> None:
        """Register backward hooks on all trainable parameters."""
        for name, param in self.model.policy.named_parameters():
            if not param.requires_grad:
                continue

            friendly = self._friendly_name(name)

            def make_hook(layer_name: str):
                def hook(grad):
                    norm = float(grad.norm().item())
                    if layer_name not in self._grad_norms:
                        self._grad_norms[layer_name] = []
                    self._grad_norms[layer_name].append(norm)
                return hook

            h = param.register_hook(make_hook(friendly))
            self._hooks.append(h)

        logger.info(f"Gradient monitoring: registered hooks on {len(self._hooks)} parameters")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0 or self.num_timesteps == 0:
            return True

        if not self._grad_norms:
            return True

        try:
            self._log_and_save()
        except Exception as e:
            logger.warning(f"Gradient norm logging failed: {e}")

        return True

    def _log_and_save(self) -> None:
        """Compute mean norms, log to console/TB, save plots, then reset."""
        step = self.num_timesteps

        # Compute mean norm per layer
        mean_norms: Dict[str, float] = {}
        for layer, norms in self._grad_norms.items():
            mean_norms[layer] = float(np.mean(norms))

        # Reset accumulator
        self._grad_norms = {}

        # Save snapshot
        self.history.append({'timestep': step, 'norms': mean_norms.copy()})

        # Log to TensorBoard
        if self.model.logger is not None:
            for layer, norm in mean_norms.items():
                self.model.logger.record(f"grad_norm/{layer}", norm)

        # Console summary (grouped by actor/critic)
        actor_norms = {k: v for k, v in mean_norms.items() if k.startswith(('actor', 'input'))}
        critic_norms = {k: v for k, v in mean_norms.items() if k.startswith(('critic',))}
        actor_head = {k: v for k, v in mean_norms.items() if k.startswith('actor_head')}
        critic_head = {k: v for k, v in mean_norms.items() if k.startswith('critic_head')}

        total_actor = sum(actor_norms.values()) + sum(actor_head.values())
        total_critic = sum(critic_norms.values()) + sum(critic_head.values())

        logger.info(f"{'='*60}")
        logger.info(f"GRADIENT NORMS @ step {step:,}")
        logger.info(f"{'='*60}")
        logger.info(f"  Actor total:  {total_actor:.6f}")
        logger.info(f"  Critic total: {total_critic:.6f}")
        logger.info(f"  Ratio (C/A):  {total_critic / max(total_actor, 1e-10):.2f}x")
        logger.info(f"  Per layer:")

        # Sort layers for consistent display
        for layer in sorted(mean_norms.keys()):
            bar_len = min(int(mean_norms[layer] * 200), 40)
            bar = '#' * bar_len
            logger.info(f"    {layer:<35s} {mean_norms[layer]:.6f} {bar}")

        # Check for dead layers (near-zero gradients)
        dead = [k for k, v in mean_norms.items() if v < 1e-7]
        if dead:
            logger.warning(f"  DEAD LAYERS (grad < 1e-7): {dead}")

        # Check for exploding gradients
        exploding = [k for k, v in mean_norms.items() if v > 10.0]
        if exploding:
            logger.warning(f"  EXPLODING GRADIENTS (grad > 10): {exploding}")

        # Save plots
        if self.log_dir:
            self._save_plots(step, mean_norms)

    def _save_plots(self, step: int, mean_norms: Dict[str, float]) -> None:
        """Save bar chart + evolution line plot."""
        grad_dir = self.log_dir / "gradient_norms"
        grad_dir.mkdir(parents=True, exist_ok=True)

        # --- Bar chart of current norms ---
        layers = sorted(mean_norms.keys())
        values = [mean_norms[l] for l in layers]

        # Color by group
        colors = []
        for l in layers:
            if l.startswith('input'):
                colors.append('#4A90D9')  # blue
            elif l.startswith('actor'):
                colors.append('#4AD98B')  # green
            elif l.startswith('critic'):
                colors.append('#D94A4A')  # red
            else:
                colors.append('#888888')

        fig, ax = plt.subplots(figsize=(10, max(4, len(layers) * 0.3)))
        y_pos = np.arange(len(layers))
        ax.barh(y_pos, values, color=colors, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layers, fontsize=7)
        ax.set_xlabel('Mean Gradient L2 Norm')
        ax.set_title(f'Per-Layer Gradient Norms — Step {step:,}')
        ax.grid(axis='x', alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_items = [
            Patch(facecolor='#4A90D9', label='Input'),
            Patch(facecolor='#4AD98B', label='Actor'),
            Patch(facecolor='#D94A4A', label='Critic'),
        ]
        ax.legend(handles=legend_items, loc='lower right', fontsize=8)

        plt.tight_layout()
        plt.savefig(grad_dir / 'gradient_norms_latest.png', dpi=120, bbox_inches='tight')
        plt.close(fig)

        # --- Evolution plot (if enough history) ---
        if len(self.history) >= 2:
            self._save_evolution_plot(grad_dir)

    def _save_evolution_plot(self, grad_dir: Path) -> None:
        """Line plot showing gradient norms over training time."""
        timesteps = [h['timestep'] for h in self.history]

        # Collect all layer names that appear in any snapshot
        all_layers = set()
        for h in self.history:
            all_layers.update(h['norms'].keys())

        # Group layers
        groups = {
            'Actor Layers': [l for l in sorted(all_layers) if l.startswith('actor') and 'head' not in l],
            'Critic Layers': [l for l in sorted(all_layers) if l.startswith('critic') and 'head' not in l],
            'Heads + Input': [l for l in sorted(all_layers) if l.startswith(('input', 'actor_head', 'critic_head'))],
        }

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, (group_name, layer_list) in zip(axes, groups.items()):
            for layer in layer_list:
                vals = [h['norms'].get(layer, 0.0) for h in self.history]
                # Short label: remove common prefix
                short = layer.split('/')[-2] + '/' + layer.split('/')[-1] if '/' in layer else layer
                ax.plot(timesteps, vals, label=short, linewidth=1.5, alpha=0.8)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Mean Gradient Norm')
            ax.set_title(group_name)
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True, alpha=0.3)
            # Format x-axis as K
            ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))

        plt.suptitle('Gradient Norm Evolution Over Training', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(grad_dir / 'gradient_evolution.png', dpi=120, bbox_inches='tight')
        plt.close(fig)

    def _on_training_end(self) -> None:
        """Remove hooks and save final snapshot."""
        # Log final norms if any accumulated
        if self._grad_norms:
            try:
                self._log_and_save()
            except Exception:
                pass

        # Remove all hooks
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        logger.info("Gradient monitoring: hooks removed")
