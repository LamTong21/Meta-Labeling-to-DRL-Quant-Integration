from pathlib import Path
from typing import Any, Dict, Optional, Type, Union
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.models.custom_policies import DirichletActorCriticPolicy, MultiHeadSimplexPolicy


class EntropyLoggingCallback(BaseCallback):
    """Monitors policy entropy and cash-weight saturation across rollout updates
    to diagnose cash-trap policy collapse early.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.entropy_history = []
        self.cash_ratio_history = []

    def _on_step(self) -> bool:
        # Collect environment info dict from the current step
        infos = self.locals.get("infos", [])
        if infos and len(infos) > 0:
            first_info = infos[0]
            if "cash_weight" in first_info:
                self.cash_ratio_history.append(first_info["cash_weight"])
        return True

    def _on_rollout_end(self) -> None:
        # Track mean cash ratio in the latest rollout
        if self.cash_ratio_history:
            recent_cash = np.mean(self.cash_ratio_history[-100:])
            self.logger.record("rollout/mean_cash_weight", float(recent_cash))


class DRLPortfolioAgent:
    """Wrapper managing training, logging, hyperparameter updates, and inference
    for on-policy PPO reinforcement learning agents within Gymnasium portfolio environments.
    """
    def __init__(
        self,
        env: gym.Env,
        policy_type: Union[str, Type[MultiHeadSimplexPolicy], Type[DirichletActorCriticPolicy]] = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seed: int = 42,
        device: str = "auto",
    ):
        self.raw_env = env
        self.vec_env = DummyVecEnv([lambda: env])

        # Resolve policy choice
        resolved_policy = policy_type
        if policy_type == "dirichlet":
            resolved_policy = DirichletActorCriticPolicy
        elif policy_type == "simplex_mlp":
            resolved_policy = MultiHeadSimplexPolicy

        self.model = PPO(
            policy=resolved_policy,
            env=self.vec_env,
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
            seed=seed,
            device=device,
            verbose=1,
        )

    def train(self, total_timesteps: int = 200_000) -> None:
        """Executes PPO optimization loops with policy entropy tracking."""
        callback = EntropyLoggingCallback()
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Outputs asset allocation weights w_t for a single environment state."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes network parameters to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, filepath: Union[str, Path]) -> None:
        """Loads pretrained network parameters from disk."""
        self.model = PPO.load(str(filepath), env=self.vec_env)