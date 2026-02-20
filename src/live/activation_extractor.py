"""
Activation Extractor for Live Trading Visualization.

Extracts intermediate activations from TCN Analyst and PPO Agent
using PyTorch forward hooks for real-time visualization.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ActivationData:
    """Container for extracted activation data."""
    # TCN Analyst
    analyst_context: List[float] = field(default_factory=list)
    analyst_5m_encoding: List[float] = field(default_factory=list)
    analyst_15m_encoding: List[float] = field(default_factory=list)
    analyst_45m_encoding: List[float] = field(default_factory=list)
    analyst_p_up: float = 0.0
    analyst_p_down: float = 0.0
    analyst_confidence: float = 0.0
    analyst_edge: float = 0.0

    # PPO Agent - Actor
    actor_layer_0: List[float] = field(default_factory=list)
    actor_layer_1: List[float] = field(default_factory=list)
    action_logits: List[float] = field(default_factory=list)
    action_probs_flat: float = 0.0
    action_probs_long: float = 0.0
    action_probs_short: float = 0.0

    # PPO Agent - Critic
    critic_layer_0: List[float] = field(default_factory=list)
    critic_layer_1: List[float] = field(default_factory=list)
    critic_layer_2: List[float] = field(default_factory=list)
    value_estimate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'analyst': {
                'context': self.analyst_context,
                'encodings': {
                    '5m': self.analyst_5m_encoding,
                    '15m': self.analyst_15m_encoding,
                    '45m': self.analyst_45m_encoding,
                },
                'probabilities': {
                    'p_up': self.analyst_p_up,
                    'p_down': self.analyst_p_down,
                },
                'confidence': self.analyst_confidence,
                'edge': self.analyst_edge,
            },
            'agent': {
                'actor_layers': {
                    'layer_0': self.actor_layer_0,
                    'layer_1': self.actor_layer_1,
                },
                'critic_layers': {
                    'layer_0': self.critic_layer_0,
                    'layer_1': self.critic_layer_1,
                    'layer_2': self.critic_layer_2,
                },
                'action_logits': self.action_logits,
                'action_probs': {
                    'flat': self.action_probs_flat,
                    'long': self.action_probs_long,
                    'short': self.action_probs_short,
                },
                'value_estimate': self.value_estimate,
            }
        }


class ActivationExtractor:
    """
    Extract intermediate activations from TCN Analyst and PPO Agent.

    Uses PyTorch forward hooks to capture hidden layer outputs
    without modifying the model code.
    """

    def __init__(self, analyst: Optional[nn.Module], agent: Any):
        """
        Initialize extractor with models.

        Args:
            analyst: TCN Analyst model (can be None if analyst is disabled)
            agent: SniperAgent instance
        """
        self.analyst = analyst
        self.agent = agent

        # Storage for captured activations
        self._actor_activations: Dict[str, torch.Tensor] = {}
        self._critic_activations: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Register hooks on PPO policy network
        self._register_hooks()

        analyst_status = "enabled" if analyst is not None else "disabled"
        logger.info(f"ActivationExtractor initialized (analyst={analyst_status})")

    def _register_hooks(self) -> None:
        """Register forward hooks on PPO policy network layers."""
        try:
            policy = self.agent.model.policy

            # Hook actor (policy) network layers
            if hasattr(policy, 'mlp_extractor') and hasattr(policy.mlp_extractor, 'policy_net'):
                policy_net = policy.mlp_extractor.policy_net
                for i, layer in enumerate(policy_net):
                    if isinstance(layer, nn.Linear):
                        hook = layer.register_forward_hook(
                            self._make_hook(f'actor_{i}', self._actor_activations)
                        )
                        self._hooks.append(hook)
                        logger.debug(f"Registered hook on actor layer {i}")

            # Hook critic (value) network layers
            if hasattr(policy, 'mlp_extractor') and hasattr(policy.mlp_extractor, 'value_net'):
                value_net = policy.mlp_extractor.value_net
                for i, layer in enumerate(value_net):
                    if isinstance(layer, nn.Linear):
                        hook = layer.register_forward_hook(
                            self._make_hook(f'critic_{i}', self._critic_activations)
                        )
                        self._hooks.append(hook)
                        logger.debug(f"Registered hook on critic layer {i}")

            logger.info(f"Registered {len(self._hooks)} forward hooks on PPO policy")

        except Exception as e:
            logger.warning(f"Could not register PPO hooks: {e}")

    def _make_hook(self, name: str, storage: Dict) -> callable:
        """Create a forward hook that stores activations."""
        def hook(module: nn.Module, input: Tuple, output: torch.Tensor):
            # Apply activation function (ReLU is typical for SB3)
            # The hook captures output BEFORE activation in Sequential
            # For post-activation, we need to apply it manually or hook after ReLU
            storage[name] = output.detach()
        return hook

    def extract_analyst_activations(
        self,
        x_5m: Optional[torch.Tensor] = None,
        x_15m: Optional[torch.Tensor] = None,
        x_45m: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Extract TCN Analyst activations.

        Args:
            x_5m: 5-minute timeframe input tensor
            x_15m: 15-minute timeframe input tensor
            x_45m: 45-minute timeframe input tensor

        Returns:
            Dictionary with context vector and per-timeframe encodings
        """
        # Return empty activations if analyst is disabled
        if self.analyst is None:
            return {
                'context': [],
                '5m_encoding': [],
                '15m_encoding': [],
                '45m_encoding': [],
            }

        with torch.no_grad():
            context, activations = self.analyst.get_activations(x_5m, x_15m, x_45m)

            return {
                'context': context.cpu().numpy().flatten().tolist(),
                '5m_encoding': activations.get('5m', torch.zeros(1)).cpu().numpy().flatten().tolist(),
                '15m_encoding': activations.get('15m', torch.zeros(1)).cpu().numpy().flatten().tolist(),
                '45m_encoding': activations.get('45m', torch.zeros(1)).cpu().numpy().flatten().tolist(),
            }

    def extract_agent_activations(
        self,
        observation: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract PPO Agent activations during prediction.

        This method performs a forward pass which triggers the hooks,
        then returns the action and captured activations.

        Args:
            observation: Current observation vector

        Returns:
            Tuple of (action, activations_dict)
        """
        # Clear previous activations
        self._actor_activations.clear()
        self._critic_activations.clear()

        # Convert observation to tensor
        obs_tensor, _ = self.agent.model.policy.obs_to_tensor(observation)

        with torch.no_grad():
            # Get action distribution (triggers forward pass and hooks)
            dist = self.agent.model.policy.get_distribution(obs_tensor)

            # Get value estimate
            value = self.agent.model.policy.predict_values(obs_tensor)

            # Extract action probabilities for direction (first 3 logits)
            if hasattr(dist, 'distribution'):
                if isinstance(dist.distribution, list):
                    # MultiDiscrete: list of Categorical distributions
                    direction_probs = dist.distribution[0].probs[0].cpu().numpy()
                else:
                    # Fallback
                    all_logits = dist.distribution.logits
                    direction_logits = all_logits[:, :3]
                    direction_probs = torch.softmax(direction_logits, dim=1)[0].cpu().numpy()
            else:
                direction_probs = np.array([0.33, 0.33, 0.34])

        # Collect actor activations
        actor_acts = {}
        layer_idx = 0
        for key in sorted(self._actor_activations.keys()):
            if 'actor' in key:
                tensor = self._actor_activations[key]
                # Apply ReLU to match actual activation
                activated = torch.relu(tensor).cpu().numpy().flatten().tolist()
                actor_acts[f'layer_{layer_idx}'] = activated
                layer_idx += 1

        # Collect critic activations
        critic_acts = {}
        layer_idx = 0
        for key in sorted(self._critic_activations.keys()):
            if 'critic' in key:
                tensor = self._critic_activations[key]
                activated = torch.relu(tensor).cpu().numpy().flatten().tolist()
                critic_acts[f'layer_{layer_idx}'] = activated
                layer_idx += 1

        return {
            'actor_layers': actor_acts,
            'critic_layers': critic_acts,
            'action_probs': {
                'flat': float(direction_probs[0]),
                'long': float(direction_probs[1]),
                'short': float(direction_probs[2]),
            },
            'value_estimate': float(value.cpu().numpy().flatten()[0]),
        }

    def extract_full_activations(
        self,
        x_5m: Optional[torch.Tensor],
        x_15m: Optional[torch.Tensor],
        x_45m: Optional[torch.Tensor],
        observation: np.ndarray,
        p_up: float,
        p_down: float
    ) -> ActivationData:
        """
        Extract all activations from both models.

        Args:
            x_5m: 5-minute timeframe input tensor
            x_15m: 15-minute timeframe input tensor
            x_45m: 45-minute timeframe input tensor
            observation: Full observation vector for agent
            p_up: Analyst probability of up move
            p_down: Analyst probability of down move

        Returns:
            ActivationData containing all extracted activations
        """
        # Extract analyst activations
        analyst_acts = self.extract_analyst_activations(x_5m, x_15m, x_45m)

        # Extract agent activations
        agent_acts = self.extract_agent_activations(observation)

        # Build ActivationData
        data = ActivationData(
            # Analyst
            analyst_context=analyst_acts['context'],
            analyst_5m_encoding=analyst_acts['5m_encoding'],
            analyst_15m_encoding=analyst_acts['15m_encoding'],
            analyst_45m_encoding=analyst_acts['45m_encoding'],
            analyst_p_up=p_up,
            analyst_p_down=p_down,
            analyst_confidence=max(p_up, p_down),
            analyst_edge=p_up - p_down,

            # Actor
            actor_layer_0=agent_acts['actor_layers'].get('layer_0', []),
            actor_layer_1=agent_acts['actor_layers'].get('layer_1', []),
            action_probs_flat=agent_acts['action_probs']['flat'],
            action_probs_long=agent_acts['action_probs']['long'],
            action_probs_short=agent_acts['action_probs']['short'],

            # Critic
            critic_layer_0=agent_acts['critic_layers'].get('layer_0', []),
            critic_layer_1=agent_acts['critic_layers'].get('layer_1', []),
            critic_layer_2=agent_acts['critic_layers'].get('layer_2', []),
            value_estimate=agent_acts['value_estimate'],
        )

        return data

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.info("Removed all forward hooks")

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()
