"""
PPO Sniper Agent wrapper using Stable Baselines 3.

Features:
- CPU-only PPO by default (stable SB3 training)
- Memory-efficient callbacks
- Training and inference methods
- Model saving/loading
"""

import os
import torch
import numpy as np
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import gc
import logging

logger = logging.getLogger(__name__)

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".mplconfig"),
)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym

from src.utils.callbacks import MemoryCleanupCallback


class EntropyScheduleCallback(BaseCallback):
    """
    Callback with 4-phase stepped entropy for training.

    v26: Extended phases for longer exploration.

    Phase 1 (0-20M):       ent_coef = 0.02  (reduced explore)
    Phase 2 (20M-300M):    ent_coef = 0.01  (transition)
    Phase 3 (300M-600M):   ent_coef = 0.005 (exploit)
    Phase 4 (600M+):       ent_coef = 0.002 (confident policies)
    """

    def __init__(
        self,
        entropy_config=None,  # EntropyScheduleConfig from settings
        phase1_steps: Optional[int] = None,
        phase2_steps: Optional[int] = None,
        phase3_steps: Optional[int] = None,
        phase1_ent: Optional[float] = None,
        phase2_ent: Optional[float] = None,
        phase3_ent: Optional[float] = None,
        phase4_ent: Optional[float] = None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        # Resolve from centralized config if not provided
        if entropy_config is None:
            from config.settings import config as default_config
            entropy_config = default_config.entropy_schedule

        self.phase1_steps = phase1_steps if phase1_steps is not None else entropy_config.phase1_steps
        self.phase2_steps = phase2_steps if phase2_steps is not None else entropy_config.phase2_steps
        self.phase3_steps = phase3_steps if phase3_steps is not None else entropy_config.phase3_steps
        self.phase1_ent = phase1_ent if phase1_ent is not None else entropy_config.phase1_ent
        self.phase2_ent = phase2_ent if phase2_ent is not None else entropy_config.phase2_ent
        self.phase3_ent = phase3_ent if phase3_ent is not None else entropy_config.phase3_ent
        self.phase4_ent = phase4_ent if phase4_ent is not None else entropy_config.phase4_ent
        self.current_phase = 1

    def _on_step(self) -> bool:
        steps = self.num_timesteps

        # Determine current phase and entropy
        if steps < self.phase1_steps:
            # Phase 1: Reduced Explore (0-20M)
            current_ent_coef = self.phase1_ent
            new_phase = 1
        elif steps < self.phase1_steps + self.phase2_steps:
            # Phase 2: Transition (20M-300M)
            current_ent_coef = self.phase2_ent
            new_phase = 2
        elif steps < self.phase1_steps + self.phase2_steps + self.phase3_steps:
            # Phase 3: Exploit (300M-600M)
            current_ent_coef = self.phase3_ent
            new_phase = 3
        else:
            # Phase 4: Confident (600M+)
            current_ent_coef = self.phase4_ent
            new_phase = 4

        # Update the model's entropy coefficient
        self.model.ent_coef = current_ent_coef

        # Log phase transitions
        if new_phase != self.current_phase:
            phase_names = {1: "EXPLORE", 2: "TRANSITION", 3: "EXPLOIT", 4: "CONFIDENT"}
            logger.info(f"[EntropySchedule] Phase {new_phase} ({phase_names[new_phase]}): ent_coef = {current_ent_coef}")
            self.current_phase = new_phase

        return True


class TrainingMetricsCallback(BaseCallback):
    """
    Callback to log training metrics.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_trades_history = []  # Track trades per episode

    def _on_step(self) -> bool:
        # Check for episode end
        if self.locals.get('dones') is not None:
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][idx]
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_lengths.append(info['episode']['l'])
                        
                        # Capture n_trades if available
                        n_trades = info.get('n_trades', 0)
                        self.episode_trades_history.append(n_trades)

        # Log periodically
        if self.n_calls % self.log_freq == 0 and self.verbose > 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                mean_trades = np.mean(self.episode_trades_history[-100:]) if self.episode_trades_history else 0.0

                logger.info(f"Step {self.n_calls}: Mean Reward={mean_reward:.2f}, "
                            f"Mean Length={mean_length:.0f}, Mean Trades={mean_trades:.1f}")

        return True


class SniperAgent:
    """
    PPO-based Sniper Agent for the trading environment.

    Wraps Stable Baselines 3 PPO with:
    - CPU device selection (default)
    - Custom network architecture
    - Memory-efficient training
    - Evaluation and inference methods
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 256,        # Increased from 64 for stability
        n_epochs: int = 4,            # Match config/settings.py (was 20, caused overfitting)
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.02,       # Reduced exploration baseline
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        net_arch: Optional[list] = None,
        device: Optional[str | torch.device] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
        tensorboard_log: Optional[str] = None,
    ):
        """
        Initialize the Sniper Agent.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate for PPO
            n_steps: Number of steps per update
            batch_size: Minibatch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            net_arch: Network architecture [hidden_sizes]
            device: 'mps', 'cuda', 'cpu', or None for auto
            verbose: Verbosity level
            seed: Random seed
            tensorboard_log: Directory for TensorBoard logs (None = disabled)
        """
        self.env = env
        self.verbose = verbose

        # Network architecture
        # v25: Asymmetric architecture - Critic larger than Actor
        # Value estimation (continuous scalar) is harder than action selection (3 choices)
        # Larger Critic should improve Explained Variance (was ~0.20, target >0.5)
        if net_arch is None:
            net_arch = [256, 256]  # Default Actor size

        # Asymmetric: Actor [256,256], Critic [512,512,256]
        policy_kwargs = {
            'net_arch': dict(
                pi=net_arch,              # Actor: 2 layers of 256
                vf=[512, 512, 256]        # Critic: 3 layers, wider
            )
        }

        # Device selection with fallback
        if device is None:
            device = self._select_device()
        else:
            # Normalize to string for consistent handling.
            device_type = device if isinstance(device, str) else device.type
            if device_type != "cpu":
                if verbose > 0:
                    logger.debug(f"Overriding PPO device {device_type} -> cpu")
                device = "cpu"

        # Create PPO model
        
        try:
            self.model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=verbose,
                seed=seed,
                tensorboard_log=tensorboard_log,
            )
            self.device = device
            if verbose > 0:
                logger.debug(f"SniperAgent initialized on device: {device}")

        except Exception as e:
            # Fallback to CPU if MPS fails
            if device != 'cpu':
                logger.warning(f"Failed to use {device}, falling back to CPU: {e}")
                self.model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=learning_rate,  # Use scheduled LR, not raw value
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    policy_kwargs=policy_kwargs,
                    device='cpu',
                    verbose=verbose,
                    seed=seed,
                    tensorboard_log=tensorboard_log,
                )
                self.device = 'cpu'
            else:
                raise

    def _select_device(self) -> str:
        """Select the best available device."""
        # NOTE: SB3 PPO is most stable on CPU (especially on Apple Silicon).
        return "cpu"

    def train(
        self,
        total_timesteps: int = 500_000,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10_000,
        save_path: Optional[str] = None,
        checkpoint_save_freq: Optional[int] = None,
        callbacks: Optional[list] = None,
        callback: Optional[BaseCallback] = None,
        reset_num_timesteps: bool = True
    ) -> Dict[str, Any]:
        """
        Train the agent.

        Args:
            total_timesteps: Total training timesteps
            eval_env: Optional evaluation environment
            eval_freq: Evaluation frequency
            save_path: Path to save best model
            checkpoint_save_freq: Frequency for checkpoint saves (None uses default)
            callbacks: Additional callbacks (list)
            callback: Single callback (for convenience)
            reset_num_timesteps: Whether to reset the current timestep count (False for resuming)

        Returns:
            Training info dictionary
        """
        ppo_log_dir = None
        if save_path:
            from stable_baselines3.common.logger import configure
            from datetime import datetime

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            ppo_log_dir = Path(save_path) / "ppo_logs" / run_id
            ppo_log_dir.mkdir(parents=True, exist_ok=True)
            # Persist PPO training metrics (incl. loss) for offline inspection.
            # `progress.csv` contains fields like: train/loss, train/value_loss, rollout/ep_rew_mean, etc.
            self.model.set_logger(configure(str(ppo_log_dir), ["csv", "log", "tensorboard"]))
            if self.verbose > 0:
                logger.debug(f"SB3 PPO logs (incl. loss): {ppo_log_dir}")

        # Build callback list
        callback_list = [
            MemoryCleanupCallback(cleanup_freq=5000, verbose=self.verbose),
            TrainingMetricsCallback(log_freq=2000, verbose=self.verbose),
            # Entropy Schedule: Stepped phases (uses class defaults)
            # Phase 1 (0-20M): explore @ 0.02
            # Phase 2 (20M-300M): transition @ 0.01
            # Phase 3 (300M-600M): exploit @ 0.005
            # Phase 4 (600M+): confident @ 0.002
            EntropyScheduleCallback(verbose=self.verbose)
        ]

        # Add Checkpoint Callback (Save every N steps)
        if save_path:
            save_freq = 125_000 if checkpoint_save_freq is None else checkpoint_save_freq
            checkpoint_path = Path(save_path) / "checkpoints"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            if save_freq > 0:
                callback_list.append(CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(checkpoint_path),
                    name_prefix="sniper_model",
                    save_replay_buffer=False,
                    save_vecnormalize=True
                ))

        if callbacks:
            callback_list.extend(callbacks)

        # Support single callback parameter
        if callback is not None:
            callback_list.append(callback)

        # NOTE: We deliberately DO NOT use EvalCallback for model selection.
        # Using eval performance to select the "best" model causes overfitting
        # to the eval set. Instead, we save the FINAL model after all training.
        # The eval_env is only used for monitoring, not selection.
        #
        # If you want periodic eval logging (without selection), add custom callback.

        # Train
        if self.verbose > 0:
            logger.info(f"Starting training for {total_timesteps:,} timesteps...")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps
        )

        # Save final model
        if save_path is not None:
            self.save(Path(save_path) / "final_model")

        return {
            'total_timesteps': total_timesteps,
            'device': self.device,
            'ppo_log_dir': str(ppo_log_dir) if ppo_log_dir is not None else None,
        }

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        min_action_confidence: float = 0.0
    ) -> tuple:
        """
        Predict action for given observation with optional confidence threshold.

        Args:
            observation: Current observation
            deterministic: Use deterministic policy
            min_action_confidence: Minimum probability required to take a non-flat action.
                                 If confidence < threshold, action is forced to Flat (0).
                                 Only applies to Direction (action[0]).

        Returns:
            Tuple of (action, states)
        """
        # Standard prediction
        action, states = self.model.predict(observation, deterministic=deterministic)

        # Apply confidence thresholding if requested
        if min_action_confidence > 0.0:
            # We need to get the probabilities from the policy
            # Convert observation to tensor
            obs_tensor, _ = self.model.policy.obs_to_tensor(observation)
            
            # Get distribution
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
            
            # Calculate probabilities from logits
            # SB3 MultiCategoricalDistribution stores logits in a specific way.
            # For MultiDiscrete, we usually have a list of Categorical distributions
            # or concatenated logits.
            
            # Helper to get probs for the first dimension (Direction)
            # The action space is MultiDiscrete.
            # dist.distribution is usually a list of Categorical distributions
            # IF using Independent(OneHotCategorical) or similar.
            
            # Accessing logits/probs directly depends on SB3 implementation details.
            # A robust way is to inspect `dist.distribution.probs` if available, 
            # or `dist.distribution` params.
            
            # For MultiDiscrete, SB3 often flattens the logits.
            # We know Direction is the first component.
            # Let's assume standard SB3 implementation for MultiDiscrete.
            
            # Safe access to probabilities for the Direction component (index 0)
            try:
                # Check if dist.distribution is a list (SB3 MultiDiscrete behavior)
                if isinstance(dist.distribution, list):
                    # Index 0 is Direction, Index 1 is Size
                    direction_dist = dist.distribution[0]
                    direction_probs = direction_dist.probs # Shape: (batch_size, 3)
                else:
                    # Fallback for other potential structures
                    # Try to access logits directly if not a list
                    all_logits = dist.distribution.logits
                    direction_logits = all_logits[:, :3]
                    direction_probs = torch.softmax(direction_logits, dim=1)

                # Get confidence of the CHOSEN action for Direction
                # Check if vectorized (batch size > 1) or single
                if len(action.shape) == 1:
                    # Single environment, action is [dir, size, ...]
                    chosen_dir = action[0]
                    confidence = direction_probs[0, chosen_dir].item()
                    
                    if confidence < min_action_confidence and chosen_dir != 0:
                        # DEBUG: Print intervention
                        # print(f"THRESHOLD INTERVENTION: Action {chosen_dir} (Conf {confidence:.2f} < {min_action_confidence}) -> FLAT")
                        # Force Flat
                        action[0] = 0
                        
                else:
                    # Vectorized environments (n_envs, n_actions)
                    # Iterate over envs
                    for i in range(len(action)):
                        chosen_dir = action[i, 0]
                        confidence = direction_probs[i, chosen_dir].item()
                        
                        if confidence < min_action_confidence and chosen_dir != 0:
                            # Force Flat
                            action[i, 0] = 0
                            
            except Exception as e:
                # v40 FIX: Log instead of silently ignoring
                logger.debug(f"Confidence threshold check failed: {e}")

        return action, states

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the agent on an environment.

        Args:
            env: Evaluation environment
            n_episodes: Number of episodes
            deterministic: Use deterministic policy

        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        episode_pnls = []
        episode_trades = []
        episode_win_rates = []

        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0

            while not done and not truncated:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if 'total_pnl' in info:
                episode_pnls.append(info['total_pnl'])

            # CRITICAL FIX: Track trade count and win rate
            n_trades = info.get('n_trades', 0)
            episode_trades.append(n_trades)

            win_rate = 0.0
            if n_trades > 0 and 'trades' in info:
                wins = sum(1 for t in info['trades'] if t.get('pnl', 0) > 0)
                win_rate = wins / n_trades
            episode_win_rates.append(win_rate)

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_pnl': np.mean(episode_pnls) if episode_pnls else 0.0,
            'mean_trades': np.mean(episode_trades) if episode_trades else 0.0,
            'win_rate': np.mean(episode_win_rates) if episode_win_rates else 0.0,
            'n_episodes': n_episodes
        }

    def save(self, path: str | Path):
        """Save the model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        if self.verbose > 0:
            logger.info(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: gym.Env,
        device: Optional[str | torch.device] = None
    ) -> 'SniperAgent':
        """
        Load a saved model.

        Args:
            path: Path to saved model
            env: Environment for the agent
            device: Device to load onto

        Returns:
            Loaded SniperAgent
        """
        agent = cls.__new__(cls)
        agent.env = env
        agent.verbose = 1

        if device is None:
            device = agent._select_device()

        agent.model = PPO.load(str(path), env=env, device=device)
        agent.device = device

        return agent


def create_agent(
    env: gym.Env,
    config: Optional[object] = None,
    device: Optional[str | torch.device] = None,
    tensorboard_log: Optional[str] = None,
) -> SniperAgent:
    """
    Factory function to create SniperAgent with config.

    Args:
        env: Trading environment
        config: AgentConfig object
        device: Device for training
        tensorboard_log: Directory for TensorBoard logs

    Returns:
        SniperAgent instance
    """
    if config is None:
        return SniperAgent(env, device=device, tensorboard_log=tensorboard_log)

    return SniperAgent(
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        net_arch=config.net_arch if hasattr(config, 'net_arch') else None,
        device=device,
        tensorboard_log=tensorboard_log,
    )


