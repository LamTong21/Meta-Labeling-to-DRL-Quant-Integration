from typing import Any, Dict, List, Optional, Tuple, Type
import gymnasium as gym
import numpy as np
import torch as th
from torch import nn
from torch.distributions import Dirichlet
from stable_baselines3.common.policies import ActorCriticPolicy


class SimplexActorHead(nn.Module):
    """Maps latent feature representations into a continuous probability simplex
    Delta^{N+1} via Softmax normalization with temperature scaling.
    """
    def __init__(self, latent_dim: int, action_dim: int, temperature: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(latent_dim, action_dim)
        self.temperature = max(temperature, 1e-4)

    def forward(self, x: th.Tensor) -> th.Tensor:
        logits = self.linear(x) / self.temperature
        return th.softmax(logits, dim=-1)


class MultiHeadSimplexPolicy(ActorCriticPolicy):
    """Custom MLP Actor-Critic Architecture separating Value regression from
    Simplex-constrained asset allocation.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Any,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):
        if net_arch is None:
            net_arch = dict(pi=[256, 128], vf=[256, 128])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        super()._build_mlp_extractor()

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass outputting action simplex allocations, values, and log-probabilities."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob


class DirichletActorCriticPolicy(ActorCriticPolicy):
    """Actor-Critic network parameterizing a Dirichlet distribution over portfolio allocations.
    
    Directly models positive concentrations alpha > 0 over the unit simplex:
        a_t ~ Dirichlet(alpha_1, ..., alpha_{N+1})
    Eliminates cash-absorbing boundary collapses by bounding entropy degradation explicitly:
        alpha_i = softplus(z_i) + eps
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Any,
        concentration_eps: float = 1.01,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        self.concentration_eps = concentration_eps
        action_dim = action_space.shape[0]

        # Replace standard Gaussian policy head with positive concentration projection
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.concentration_net = nn.Sequential(
            nn.Linear(latent_dim_pi, action_dim),
            nn.Softplus(),
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        values = self.value_net(latent_vf)
        concentrations = self.concentration_net(latent_pi) + self.concentration_eps
        
        dirichlet_dist = Dirichlet(concentrations)

        if deterministic:
            # Mode / Mean allocation under exploitation
            actions = concentrations / th.sum(concentrations, dim=-1, keepdim=True)
        else:
            actions = dirichlet_dist.rsample()

        log_prob = dirichlet_dist.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        values = self.value_net(latent_vf)
        concentrations = self.concentration_net(latent_pi) + self.concentration_eps
        dirichlet_dist = Dirichlet(concentrations)

        # Boundary safeguard for numerical zero inputs in log calculations
        actions_safe = th.clamp(actions, min=1e-6, max=1.0 - 1e-6)
        actions_safe = actions_safe / th.sum(actions_safe, dim=-1, keepdim=True)

        log_prob = dirichlet_dist.log_prob(actions_safe)
        entropy = dirichlet_dist.entropy()

        return values, log_prob, entropy