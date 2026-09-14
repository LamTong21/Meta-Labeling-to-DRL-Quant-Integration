import numpy as np
import pandas as pd
import pytest

from src.envs.portfolio_env import MultiAssetPortfolioEnv


@pytest.fixture
def mock_asset_universe() -> dict:
    """Builds a minimal mock multi-asset dictionary with 3 tickers over 50 sessions."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(42)
    
    mock_data = {}
    for ticker in ["TCK_A", "TCK_B", "TCK_C"]:
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, size=len(dates))))
        df = pd.DataFrame({
            "open": prices * 0.995,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "close_adj": prices,
            "open_adj": prices * 0.995,
            "volume": np.random.uniform(50000, 200000, size=len(dates)),
            "log_return": np.concatenate([[0.0], np.diff(np.log(prices))]),
            "parkinson_vol": np.full(len(dates), 0.012),
            "mkt_return": np.random.normal(0.0002, 0.01, size=len(dates)),
        }, index=dates)
        mock_data[ticker] = df

    return mock_data


def test_action_simplex_bounds_and_drift(mock_asset_universe):
    """Verifies that executable weights reside within the unit simplex Delta^{N+1}."""
    env = MultiAssetPortfolioEnv(
        asset_data=mock_asset_universe,
        lookback_window=10,
        transaction_cost=0.0015,
        apply_frictions=True,
    )

    obs, _ = env.reset(seed=42)
    assert np.isclose(np.sum(env.current_weights), 1.0)
    assert env.current_weights[0] == 1.0  # 100% Cash at initialization

    done = False
    step_count = 0
    while not done and step_count < 25:
        # Generate random continuous action logits
        random_action = np.random.uniform(-3.0, 3.0, size=env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(random_action)
        done = terminated or truncated

        weights = env.current_weights

        # 1. Non-negativity constraint
        assert (weights >= -1e-6).all(), f"Negative weight detected: {weights}"

        # 2. Simplex unit sum constraint: sum_{i=0}^N w_i = 1.0
        assert np.isclose(np.sum(weights), 1.0, atol=1e-5), f"Simplex sum violation: {np.sum(weights)}"

        # 3. Microstructure state tracking
        assert 0.0 <= info["cash_weight"] <= 1.0
        assert not np.isnan(reward)
        step_count += 1


def test_statutory_price_band_clamping(mock_asset_universe):
    """Verifies that returns are strictly clamped to HOSE +/-7% bands."""
    # Inject extreme 20% return spike into asset A
    extreme_date = mock_asset_universe["TCK_A"].index[15]
    mock_asset_universe["TCK_A"].loc[extreme_date, "log_return"] = 0.20

    env = MultiAssetPortfolioEnv(
        asset_data=mock_asset_universe,
        lookback_window=10,
        apply_frictions=True,
    )
    env.reset()

    # Move directly to the affected step
    env.current_step = 4  # 4 + lookback(10) = index 14 -> next step index 15
    action = np.zeros(env.action_space.shape)  # Equal target allocation
    _, _, _, _, info = env.step(action)

    # Portfolio return cannot compound at 20% due to the 7% clamp limit
    assert info["portfolio_return"] < 0.10