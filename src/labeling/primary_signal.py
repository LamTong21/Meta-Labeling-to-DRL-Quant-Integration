from typing import Literal
import numpy as np
import pandas as pd


def generate_primary_signal(
    df: pd.DataFrame,
    fast_window: int = 10,
    slow_window: int = 30,
    price_col: str = "close_adj",
    method: Literal["ma_crossover", "momentum", "breakout"] = "ma_crossover",
) -> pd.Series:
    """Generates directional hypothesis propositions: f(X_t) -> y_hat_t in {-1, 0, 1}[cite: 1].
    
    Acts as the base directional trigger before applying meta-labeling[cite: 1].
    """
    price = df[price_col]
    signals = pd.Series(0, index=df.index, name="primary_signal")

    if method == "ma_crossover":
        fast_ma = price.rolling(window=fast_window, min_periods=fast_window).mean()
        slow_ma = price.rolling(window=slow_window, min_periods=slow_window).mean()
        
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1

    elif method == "momentum":
        returns_rolling = price.pct_change(periods=fast_window)
        signals[returns_rolling > 0] = 1
        signals[returns_rolling < 0] = -1

    elif method == "breakout":
        upper_channel = price.rolling(window=slow_window).max().shift(1)
        lower_channel = price.rolling(window=slow_window).min().shift(1)
        
        signals[price > upper_channel] = 1
        signals[price < lower_channel] = -1

    else:
        raise ValueError(f"Unknown signal generation method: {method}")

    # Forward-fill intermediate positions and remove indeterminate starting sessions
    signals = signals.ffill().fillna(0).astype(int)
    return signals