from typing import Dict, Optional
import numpy as np
import pandas as pd


def raw_data_profile(df_raw: pd.DataFrame, symbol: str) -> None:
    """Verifies physical bar geometry consistency and prints missing value ledger[cite: 1, 2]."""
    print("\n" + "=" * 80)[cite: 1, 2]
    print("TỔNG QUAN DATASET KẾT HỢP (DAILY + MICROSTRUCTURE):")[cite: 1, 2]
    print(f"- Số dòng tổng cộng      : {len(df_raw):,}")[cite: 1, 2]
    print(f"- Mã cổ phiếu            : {symbol}")[cite: 1, 2]
    print(f"- Khung thời gian        : Từ {df_raw.index.min().date()} đến {df_raw.index.max().date()}")[cite: 1, 2]
    print(f"- Tổng giá trị khuyết NaN: {df_raw.isna().sum().sum()}")[cite: 1, 2]
    print("=" * 80)[cite: 1, 2]

    # Verify physical boundary condition: Low <= min(O,C) and High >= max(O,C)
    min_oc = df_raw[["open", "close"]].min(axis=1)[cite: 1, 2]
    max_oc = df_raw[["open", "close"]].max(axis=1)[cite: 1, 2]
    low_violations = np.sum(df_raw["low"] > min_oc)
    high_violations = np.sum(df_raw["high"] < max_oc)

    print(f"Candlestick Geometry Check: Low Violations = {low_violations}, High Violations = {high_violations}")

    nan_counts = df_raw.isna().sum()
    isolated_nans = nan_counts[nan_counts > 0]
    if not isolated_nans.empty:
        print("\n[Chi tiết biến khuyết (NaN Ledger)]:")
        print(isolated_nans.to_string())
    else:
        print("\nDữ liệu hoàn hảo, không có giá trị khuyết thiếu!")[cite: 1, 2]


def evaluate_missingness_ledger(asset_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Builds universe-wide Missing Value Ledger to detect unhandled drops or data holes."""
    records = []
    for ticker, df in asset_dict.items():
        total_nans = int(df.isna().sum().sum())[cite: 1, 2]
        nan_cols = df.isna().sum()[df.isna().sum() > 0]
        details = ", ".join([f"{col} ({count})" for col, count in nan_cols.items()]) if total_nans > 0 else "None"

        records.append({
            "Ticker": ticker,
            "Session Count": len(df),
            "Total NaNs": total_nans,
            "Localized Features of Missingness": details,
        })

    ledger_df = pd.DataFrame(records).sort_values(by="Ticker")
    return ledger_df


def compute_policy_entropy(action_probabilities: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Computes Shannon entropy across the action simplex to diagnose cash-trap collapse:
        H(pi_theta(cdot | s_t)) = - sum_{i=0}^N w_i * ln(w_i).
    """
    probs_safe = np.clip(action_probabilities, eps, 1.0)
    entropy_per_step = -np.sum(probs_safe * np.log(probs_safe), axis=-1)
    return entropy_per_step