import numpy as np
import pandas as pd


def decompose_returns_and_volatility(
    df: pd.DataFrame, 
    max_calendar_gap_days: int = 7
) -> pd.DataFrame:
    """Computes compounding return components and extremum volatility proxy[cite: 1, 2].
    
    Formulations:
        r_{i,t} = ln(C_t^{adj} / C_{t-1}^{adj})[cite: 1, 2]
        r_{i,t}^{overnight} = ln(O_t^{adj} / C_{t-1}^{adj})[cite: 1, 2]
        r_{i,t}^{intraday} = ln(C_t^{adj} / O_t^{adj})[cite: 1, 2]
        sigma_{i,t}^{Parkinson} = sqrt( (1 / (4*ln 2)) * [ln(H_t^{adj} / L_t^{adj})]^2 )
    
    Masks suspension gaps (delta_tau > 7 calendar days) to avoid phantom jumps[cite: 1, 2].
    """
    df = df.copy()
    
    # Calendar gap detection
    time_diff = df.index.to_series().diff().dt.days[cite: 1, 2]
    
    # Continuous compounding returns
    df["log_return"] = np.log(df["close_adj"] / df["close_adj"].shift(1))[cite: 1, 2]
    df["overnight_return"] = np.log(df["open_adj"] / df["close_adj"].shift(1))[cite: 1, 2]
    df["intraday_return"] = np.log(df["close_adj"] / df["open_adj"])[cite: 1, 2]
    df["hl_log_range"] = np.log(df["high_adj"] / df["low_adj"])[cite: 1, 2]
    
    # Parkinson volatility proxy
    inv_four_ln2 = 1.0 / (4.0 * np.log(2.0))
    df["parkinson_vol"] = np.sqrt(inv_four_ln2 * (df["hl_log_range"] ** 2))
    
    # Suspension mask
    suspension_mask = time_diff > max_calendar_gap_days[cite: 1, 2]
    df.loc[suspension_mask, ["log_return", "overnight_return"]] = np.nan[cite: 1, 2]
    
    return df