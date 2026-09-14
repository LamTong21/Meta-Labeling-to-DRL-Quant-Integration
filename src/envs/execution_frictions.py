from typing import Tuple
import numpy as np


class HOSEExecutionFrictionModel:
    """Simulates emerging market microstructure frictions specific to the Ho Chi Minh Stock Exchange (HOSE):
    1. Asymmetric Statutory Price Bands: Daily price movements strictly clamped within +/- 7%.
    2. T+1.5 Settlement Latency: Lock inventory purchased on day t until the afternoon of t+1 / morning of t+2.
    """
    def __init__(
        self,
        num_assets: int,
        price_band_limit: float = 0.07,
        transaction_cost: float = 0.0015,  # 15 bps (brokerage + exchange tax + spread)
    ):
        self.num_assets = num_assets
        self.price_band_limit = price_band_limit
        self.transaction_cost = transaction_cost

        # Inventory lock pipeline tracking locked capital proportions across settlement intervals
        # Index 0: Locked today (t), Available on t+2
        # Index 1: Locked yesterday (t-1), Becomes liquid afternoon of t+1 / morning t+2
        self.locked_inventory = np.zeros((2, num_assets), dtype=np.float32)

    def reset(self):
        """Clears inventory queues upon episode initialization."""
        self.locked_inventory = np.zeros((2, self.num_assets), dtype=np.float32)

    def clamp_returns_to_bands(self, raw_returns: np.ndarray) -> np.ndarray:
        """Clamps daily returns within statutory exchange bands:
            r_clamped in [ln(1 - 0.07), ln(1 + 0.07)]
        """
        lower_bound = np.log(1.0 - self.price_band_limit)
        upper_bound = np.log(1.0 + self.price_band_limit)
        return np.clip(raw_returns, lower_bound, upper_bound)

    def apply_settlement_lock(
        self, 
        target_weights: np.ndarray, 
        current_weights: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Enforces T+1.5 liquidity constraints. An agent cannot re-sell assets 
        that have not completed the settlement cycle.
        
        Args:
            target_weights: Desired continuous weights w_t in R^{N+1} (index 0 is cash).
            current_weights: Drifted weights w'_t in R^{N+1}.

        Returns:
            executable_weights: Feasible weight vector satisfying settlement locks.
            turnover_cost: Incurred rebalancing transaction penalty.
        """
        executable_weights = target_weights.copy()
        risky_target = executable_weights[1:]
        risky_current = current_weights[1:]

        # Minimum locked positions that cannot be sold
        mandatory_holdings = np.sum(self.locked_inventory, axis=0)

        # Truncate sell orders if target falls below locked threshold
        clipped_risky = np.maximum(risky_target, mandatory_holdings)
        executable_weights[1:] = clipped_risky

        # Re-normalize over cash to maintain unit simplex sum(w) = 1
        excess_allocated = np.sum(executable_weights[1:])
        if excess_allocated > 1.0:
            # If mandatory holdings exceed 100% equity, scale down liquid portions
            executable_weights[1:] = executable_weights[1:] / excess_allocated
            executable_weights[0] = 0.0
        else:
            executable_weights[0] = 1.0 - excess_allocated

        # Track turnover and transaction friction: c_trans * ||w_t - w'_t||_1
        weight_delta = np.abs(executable_weights[1:] - risky_current)
        turnover_cost = float(self.transaction_cost * np.sum(weight_delta))

        # Advance settlement queue: newly bought equity gets locked for t+2
        buy_amounts = np.maximum(0.0, executable_weights[1:] - risky_current)
        self.locked_inventory[1] = self.locked_inventory[0]
        self.locked_inventory[0] = buy_amounts

        return executable_weights, turnover_cost