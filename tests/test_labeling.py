import numpy as np
import pandas as pd
import pytest

from src.labeling.calibrator import ProbabilityCalibrator
from src.labeling.triple_barrier import apply_triple_barrier_labeling


@pytest.fixture
def synthetic_price_path() -> Tuple[pd.DataFrame, pd.Series]:
    """Generates synthetic price trajectories with known upper/lower barrier breaches."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    
    # Path 1 (0 -> 4): Rallies from 100 to 110 (Hits upper barrier)
    # Path 2 (5 -> 9): Drops from 100 to 88 (Hits lower barrier)
    prices = [100.0, 102.0, 105.0, 110.0, 112.0, 100.0, 97.0, 93.0, 88.0, 85.0]
    
    df = pd.DataFrame({
        "close_adj": prices,
        "parkinson_vol": [0.02] * 10,
    }, index=dates)

    # Primary signal: Long at t=0, Long at t=5
    signals = pd.Series(0, index=dates)
    signals.iloc[0] = 1
    signals.iloc[5] = 1

    return df, signals


def test_triple_barrier_hits(synthetic_price_path):
    """Verifies correct meta-label assignment for profit-take vs stop-loss events."""
    df, signals = synthetic_price_path
    
    labeled_df = apply_triple_barrier_labeling(
        df=df,
        primary_signals=signals,
        volatility_col="parkinson_vol",
        price_col="close_adj",
        horizon=4,
        pt_sl_ratio=1.5,  # Barrier = max(1.5 * 0.02, 0.005) = 3%
        min_ret=0.005,
    )

    # Event 0: Rallying trajectory -> Should hit Upper Barrier first (Label = 1)
    assert labeled_df.loc[df.index[0], "meta_label"] == 1.0
    assert labeled_df.loc[df.index[0], "return_at_touch"] > 0

    # Event 5: Collapsing trajectory -> Should hit Lower Barrier first (Label = 0)
    assert labeled_df.loc[df.index[5], "meta_label"] == 0.0
    assert labeled_df.loc[df.index[5], "return_at_touch"] < 0


def test_probability_calibrator_monotonicity():
    """Verifies that Isotonic calibration produces monotonic probabilities."""
    y_raw_probs = np.array([0.1, 0.35, 0.30, 0.60, 0.55, 0.85, 0.90])
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])

    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit(y_raw_probs, y_true)
    calibrated = calibrator.predict_proba(np.sort(y_raw_probs))

    # Calibrated probabilities must be non-decreasing
    assert np.all(np.diff(calibrated) >= -1e-8)
    assert (calibrated >= 0.0).all() and (calibrated <= 1.0).all()