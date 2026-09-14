import numpy as np


def compute_downside_semi_variance_reward(
    portfolio_return: float,
    turnover_cost: float,
    lambda_penalty: float = 2.0,
    var_threshold: float = -0.02,
) -> float:
    """Calculates risk-adjusted continuous step reward R_t.

    Mathematical formulation:
        R_t = ln(1 + w_t^T r_t - Cost_turnover(w_t, w'_{t-1})) - lambda * (min(0, w_t^T r_t - VaR_alpha))^2
    """
    net_growth = 1.0 + portfolio_return - turnover_cost

    # Protective lower clip to avoid log domain collapse under liquidation scenarios
    net_growth_clipped = max(net_growth, 1e-6)
    base_utility = np.log(net_growth_clipped)

    # Downside semi-variance penalty below VaR threshold
    tail_risk = min(0.0, portfolio_return - var_threshold)
    downside_penalty = lambda_penalty * (tail_risk ** 2)

    reward = base_utility - downside_penalty
    return float(reward)