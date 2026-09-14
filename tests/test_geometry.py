import numpy as np
import pandas as pd
import pytest

from src.data.geometry_auditor import audit_candlestick_geometry, normalize_corporate_actions
from src.data.return_decomposer import decompose_returns_and_volatility


@pytest.fixture
def anomalous_bar_data() -> pd.DataFrame:
    """Creates a sample OHLCV dataset with intentional geometric and calendar anomalies."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    data = {
        # Bar 0: High is lower than max(Open, Close) -> Geometry violation
        # Bar 1: Low is higher than min(Open, Close) -> Geometry violation
        # Bar 2: Low <= 0 -> Domain violation
        # Bar 3: Volume < 0 -> Domain violation
        # Bar 4: Normal bar
        # Bar 5: Calendar gap of 10 days ahead
        "open": [100.0, 105.0, 90.0, 95.0, 100.0, 102.0],
        "high": [98.0, 110.0, 95.0, 100.0, 105.0, 104.0],
        "low": [95.0, 106.0, 0.0, 92.0, 98.0, 101.0],
        "close": [102.0, 104.0, 92.0, 96.0, 102.0, 103.0],
        "volume": [1000, 1500, 2000, -50, 1200, 800],
    }
    df = pd.DataFrame(data, index=dates)
    # Inject deliberate 10-day trading gap between bar 4 and bar 5
    new_index = list(dates[:5]) + [dates[4] + pd.Timedelta(days=10)]
    df.index = pd.DatetimeIndex(new_index)
    return df


def test_candlestick_geometry_enforcement(anomalous_bar_data):
    """Verifies that High >= max(Open, Close) and Low <= min(Open, Close) after auditing."""
    audited = audit_candlestick_geometry(anomalous_bar_data)

    # Bars with non-positive low or negative volume must be dropped
    assert len(audited) == 4
    assert (audited["low"] > 0).all()
    assert (audited["volume"] >= 0).all()

    # Geometry bounds check
    max_oc = audited[["open", "close"]].max(axis=1)
    min_oc = audited[["open", "close"]].min(axis=1)

    assert (audited["high"] >= max_oc).all()
    assert (audited["low"] <= min_oc).all()


def test_corporate_action_proportionality():
    """Verifies kappa normalization across open, high, and low series."""
    df = pd.DataFrame({
        "open": [100.0, 50.0],
        "high": [105.0, 52.0],
        "low": [95.0, 48.0],
        "close": [100.0, 50.0],
        "close_adj": [50.0, 50.0],  # 2:1 stock split adjustment on bar 0
    }, index=pd.date_range("2024-01-01", periods=2, freq="D"))

    adj_df = normalize_corporate_actions(df)
    assert np.isclose(adj_df.loc[df.index[0], "open_adj"], 50.0)
    assert np.isclose(adj_df.loc[df.index[0], "high_adj"], 52.5)
    assert np.isclose(adj_df.loc[df.index[0], "low_adj"], 47.5)


def test_calendar_gap_masking(anomalous_bar_data):
    """Verifies that returns exceeding the 7-day calendar gap threshold are masked as NaN."""
    audited = audit_candlestick_geometry(anomalous_bar_data)
    normalized = normalize_corporate_actions(audited)
    decomposed = decompose_returns_and_volatility(normalized, max_calendar_gap_days=7)

    # Bar 5 follows a 10-day gap -> log_return and overnight_return must be NaN
    assert np.isnan(decomposed["log_return"].iloc[-1])
    assert np.isnan(decomposed["overnight_return"].iloc[-1])