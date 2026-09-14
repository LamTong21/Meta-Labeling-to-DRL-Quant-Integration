import numpy as np
import pandas as pd


def audit_candlestick_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Enforces strict physical candlestick boundaries on OHLCV bars[cite: 1, 2].
    
    Mathematical constraints:
        \\tilde{H}_{i,t} = max(H_{i,t}, O_{i,t}, C_{i,t})
        \\tilde{L}_{i,t} = min(L_{i,t}, O_{i,t}, C_{i,t})
        Domain: \\tilde{L}_{i,t} > 0 and V_{i,t} >= 0[cite: 1, 2]
    """
    df = df.copy()
    
    max_oc = df[["open", "close"]].max(axis=1)[cite: 1, 2]
    min_oc = df[["open", "close"]].min(axis=1)[cite: 1, 2]
    
    # Boundary clamping
    df["high"] = np.maximum(df["high"], max_oc)[cite: 1, 2]
    df["low"] = np.minimum(df["low"], min_oc)[cite: 1, 2]
    
    # Strict physical domain filtering
    valid_mask = (df["low"] > 0) & (df["volume"] >= 0)[cite: 1, 2]
    return df[valid_mask]


def normalize_corporate_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes open, high, and low series using dividend/split factor ratio[cite: 1, 2]:
        kappa_{i,t} = C_{i,t}^{adj} / C_{i,t}[cite: 1, 2]
    """
    df = df.copy()
    if "close_adj" not in df.columns:
        df["close_adj"] = df["close"][cite: 1, 2]

    ratio = df["close_adj"] / df["close"][cite: 1, 2]
    df["open_adj"] = df["open"] * ratio[cite: 1, 2]
    df["high_adj"] = df["high"] * ratio[cite: 1, 2]
    df["low_adj"] = df["low"] * ratio[cite: 1, 2]
    
    return df