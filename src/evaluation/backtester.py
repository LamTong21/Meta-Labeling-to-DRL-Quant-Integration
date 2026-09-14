from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.envs.execution_frictions import HOSEExecutionFrictionModel


class PortfolioBacktester:
    """Deterministic, unified evaluation engine supporting both:
    1. Discrete-Event allocation runs (Modeling09 Meta-Labeling + Bet Sizing).
    2. Continuous trajectory rollouts (Modeling10 Actor-Critic DRL).
    """
    def __init__(
        self,
        asset_price_dict: Dict[str, pd.DataFrame],
        transaction_cost: float = 0.0015,
        price_band_limit: float = 0.07,
        apply_frictions: bool = True,
    ):
        self.tickers = sorted(list(asset_price_dict.keys()))
        self.num_assets = len(self.tickers)
        self.transaction_cost = transaction_cost
        self.price_band_limit = price_band_limit
        self.apply_frictions = apply_frictions

        # Intersect unified trading calendar dates across constituents
        common_idx = asset_price_dict[self.tickers[0]].index
        for t in self.tickers[1:]:
            common_idx = common_idx.intersection(asset_price_dict[t].index)
        self.dates = common_idx.sort_values()

        # Build clean chronological log-return tensor: shape (T, N)
        self.returns_df = pd.DataFrame(index=self.dates)
        for t in self.tickers:
            self.returns_df[t] = asset_price_dict[t]["log_return"].reindex(self.dates).fillna(0.0)

        self.friction_model = HOSEExecutionFrictionModel(
            num_assets=self.num_assets,
            price_band_limit=self.price_band_limit,
            transaction_cost=self.transaction_cost,
        )

    def run_weights(
        self, 
        target_weights_df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
        """Simulates historical execution over pre-computed discrete weight allocations.

        Args:
            target_weights_df: DataFrame indexed by Date, with columns matching asset tickers.
                               Remaining weight is assumed to sit in risk-free cash.

        Returns:
            net_returns: Net daily portfolio return series.
            executed_weights: Weights actually held after microstructural locks and drift.
            turnover_costs: Incurred daily transaction penalty series.
        """
        aligned_dates = self.dates.intersection(target_weights_df.index)
        T = len(aligned_dates)
        N = self.num_assets

        self.friction_model.reset()
        current_weights = np.zeros(N + 1, dtype=np.float32)
        current_weights[0] = 1.0  # Initial portfolio starts 100% in cash

        net_returns = np.zeros(T, dtype=np.float32)
        turnover_costs = np.zeros(T, dtype=np.float32)
        executed_weights_arr = np.zeros((T, N + 1), dtype=np.float32)

        for t_idx, date in enumerate(aligned_dates):
            raw_risky = target_weights_df.loc[date].to_numpy().clip(min=0.0)
            allocated_risky = np.sum(raw_risky)

            target_weights = np.zeros(N + 1, dtype=np.float32)
            if allocated_risky > 1.0:
                target_weights[1:] = raw_risky / allocated_risky
                target_weights[0] = 0.0
            else:
                target_weights[1:] = raw_risky
                target_weights[0] = 1.0 - allocated_risky

            # Apply execution frictions (T+1.5 inventory masks + fees)
            if self.apply_frictions:
                executable_weights, fee = self.friction_model.apply_settlement_lock(
                    target_weights=target_weights, 
                    current_weights=current_weights
                )
            else:
                fee = float(self.transaction_cost * np.sum(np.abs(target_weights[1:] - current_weights[1:])))
                executable_weights = target_weights

            # Realize asset returns
            daily_rets = self.returns_df.loc[date].to_numpy()
            if self.apply_frictions:
                daily_rets = self.friction_model.clamp_returns_to_bands(daily_rets)

            gross_return = float(np.sum(executable_weights[1:] * daily_rets))
            net_return = gross_return - fee

            net_returns[t_idx] = net_return
            turnover_costs[t_idx] = fee
            executed_weights_arr[t_idx] = executable_weights

            # Endogenous weight drift
            growth = np.ones(N + 1, dtype=np.float32)
            growth[1:] = 1.0 + daily_rets
            drifted = executable_weights * growth
            current_weights = drifted / np.sum(drifted)

        col_names = ["CASH"] + self.tickers
        executed_weights_df = pd.DataFrame(
            executed_weights_arr, index=aligned_dates, columns=col_names
        )
        return (
            pd.Series(net_returns, index=aligned_dates, name="net_return"),
            executed_weights_df,
            pd.Series(turnover_costs, index=aligned_dates, name="turnover_cost"),
        )