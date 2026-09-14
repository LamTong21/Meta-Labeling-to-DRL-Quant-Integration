from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.envs.execution_frictions import HOSEExecutionFrictionModel
from src.envs.reward_functions import compute_downside_semi_variance_reward


class MultiAssetPortfolioEnv(gym.Env):
    """Vectorized multi-asset portfolio rebalancing environment compliant with Gymnasium.

    State space:
        s_t = [R_{t-L:t}, Sigma_{t-L:t}^{Parkinson}, V_tilde_{t-L:t}, r_{m, t-L:t}, w_{t-1}] in R^D
    Action space:
        a_t in R^{N+1} -> Projected onto the unit probability simplex Delta^{N+1} via Softmax.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        asset_data: Dict[str, pd.DataFrame],
        lookback_window: int = 20,
        transaction_cost: float = 0.0015,
        lambda_penalty: float = 2.0,
        var_threshold: float = -0.02,
        apply_frictions: bool = True,
    ):
        super().__init__()
        self.asset_tickers = sorted(list(asset_data.keys()))
        self.num_assets = len(self.asset_tickers)
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.lambda_penalty = lambda_penalty
        self.var_threshold = var_threshold
        self.apply_frictions = apply_frictions

        # Intersect unified chronological indices across all constituents
        common_index = asset_data[self.asset_tickers[0]].index
        for t in self.asset_tickers[1:]:
            common_index = common_index.intersection(asset_data[t].index)
        self.dates = common_index.sort_values()
        self.num_steps = len(self.dates) - self.lookback_window - 1

        # Pre-align feature matrices into structured memory arrays
        # Shape: (T, N, Features)
        self._build_tensor_cache(asset_data)

        # Microstructure friction handler
        self.friction_model = HOSEExecutionFrictionModel(
            num_assets=self.num_assets,
            price_band_limit=0.07,
            transaction_cost=self.transaction_cost,
        )

        # Action Space: Raw logits for N risky assets + 1 cash component
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.num_assets + 1,), dtype=np.float32
        )

        # Observation Space: Flat concatenation of historical features and previous weight allocation
        # Features per asset = 3 (log_return, parkinson_vol, normalized_volume)
        # Macro features = 1 (mkt_return)
        # Previous weights = N + 1
        obs_dim = (
            (self.num_assets * 3 * self.lookback_window)
            + (1 * self.lookback_window)
            + (self.num_assets + 1)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.current_step = 0
        self.current_weights = np.zeros(self.num_assets + 1, dtype=np.float32)
        self.current_weights[0] = 1.0  # Initial portfolio starts 100% in cash

    def _build_tensor_cache(self, asset_data: Dict[str, pd.DataFrame]):
        """Caches tabular data into aligned tensors for vectorized step lookups."""
        T = len(self.dates)
        N = self.num_assets

        self.returns_matrix = np.zeros((T, N), dtype=np.float32)
        self.vol_matrix = np.zeros((T, N), dtype=np.float32)
        self.volume_ratio_matrix = np.zeros((T, N), dtype=np.float32)
        self.mkt_returns = np.zeros(T, dtype=np.float32)

        for i, ticker in enumerate(self.asset_tickers):
            df = asset_data[ticker].loc[self.dates]
            self.returns_matrix[:, i] = df["log_return"].fillna(0.0).to_numpy()
            self.vol_matrix[:, i] = df["parkinson_vol"].fillna(0.0).to_numpy()

            # Normalized Volume ratio: V_t / SMA_20(V)
            vol_series = df["volume"]
            sma_vol = vol_series.rolling(20, min_periods=1).mean().replace(0, 1.0)
            self.volume_ratio_matrix[:, i] = (vol_series / sma_vol).fillna(1.0).to_numpy()

        # Aligned VN-Index benchmark log return
        df_bench = asset_data[self.asset_tickers[0]].loc[self.dates]
        self.mkt_returns[:] = df_bench["mkt_return"].fillna(0.0).to_numpy()

    def _get_observation(self) -> np.ndarray:
        """Constructs observation state vector s_t at the current simulation step."""
        t_end = self.current_step + self.lookback_window
        t_start = self.current_step

        slice_ret = self.returns_matrix[t_start:t_end].T.flatten()
        slice_vol = self.vol_matrix[t_start:t_end].T.flatten()
        slice_volume_ratio = self.volume_ratio_matrix[t_start:t_end].T.flatten()
        slice_mkt = self.mkt_returns[t_start:t_end].flatten()

        state = np.concatenate(
            [slice_ret, slice_vol, slice_volume_ratio, slice_mkt, self.current_weights]
        ).astype(np.float32)
        return state

    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the simulation to step 0."""
        super().reset(seed=seed)
        self.current_step = 0
        self.current_weights = np.zeros(self.num_assets + 1, dtype=np.float32)
        self.current_weights[0] = 1.0
        self.friction_model.reset()

        observation = self._get_observation()
        info = {"date": str(self.dates[self.lookback_window])}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Transitions the continuous MDP state forward by one trading session."""
        # 1. Project action logits onto simplex Delta^{N+1} via Softmax
        exp_action = np.exp(action - np.max(action))
        target_weights = exp_action / np.sum(exp_action)

        # 2. Apply settlement locks and calculate turnover transaction friction
        if self.apply_frictions:
            executable_weights, turnover_cost = self.friction_model.apply_settlement_lock(
                target_weights=target_weights, 
                current_weights=self.current_weights
            )
        else:
            weight_delta = np.abs(target_weights[1:] - self.current_weights[1:])
            turnover_cost = float(self.transaction_cost * np.sum(weight_delta))
            executable_weights = target_weights

        # 3. Realize asset returns at session t+1
        time_idx = self.current_step + self.lookback_window
        realized_returns = self.returns_matrix[time_idx].copy()
        if self.apply_frictions:
            realized_returns = self.friction_model.clamp_returns_to_bands(realized_returns)

        # Linear portfolio return: r_port = sum_{i=1}^N w_{i,t} * r_{i,t} (Cash yields 0.0)
        portfolio_return = float(np.sum(executable_weights[1:] * realized_returns))

        # 4. Economic Reward calculation
        reward = compute_downside_semi_variance_reward(
            portfolio_return=portfolio_return,
            turnover_cost=turnover_cost,
            lambda_penalty=self.lambda_penalty,
            var_threshold=self.var_threshold,
        )

        # 5. Endogenous Weight Drift prior to active rebalancing at t+1
        growth_factors = np.ones(self.num_assets + 1, dtype=np.float32)
        growth_factors[1:] = 1.0 + realized_returns
        drifted_values = executable_weights * growth_factors
        self.current_weights = (drifted_values / np.sum(drifted_values)).astype(np.float32)

        # 6. Step progression & termination condition
        self.current_step += 1
        terminated = bool(self.current_step >= self.num_steps)
        truncated = False

        observation = self._get_observation()
        info = {
            "date": str(self.dates[time_idx]),
            "portfolio_return": portfolio_return,
            "turnover_cost": turnover_cost,
            "cash_weight": float(self.current_weights[0]),
            "max_asset_weight": float(np.max(self.current_weights[1:])),
        }

        return observation, reward, terminated, truncated, info