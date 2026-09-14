from typing import Optional
import numpy as np
import pandas as pd


def apply_triple_barrier_labeling(
    df: pd.DataFrame,
    primary_signals: pd.Series,
    volatility_col: str = "parkinson_vol",
    price_col: str = "close_adj",
    horizon: int = 5,
    pt_sl_ratio: float = 1.5,
    min_ret: float = 0.005,
) -> pd.DataFrame:
    """Implements López de Prado's path-dependent Triple-Barrier Event Sampling[cite: 1].

    Stopping time:
        tau = inf { s in (t, t+h] | |ln(P_s / P_t)| >= k * sigma_t } ^ (t+h)[cite: 1]

    Binary Meta-Label assignment:
        y_t* = 1 if (P_tau / P_t) * y_hat_t > 1 + k * sigma_t (Profit-target hit first)[cite: 1]
        y_t* = 0 otherwise (Stop-loss hit or vertical horizon expired without sufficient margin)[cite: 1]
    """
    prices = df[price_col].to_numpy()
    volatility = df[volatility_col].to_numpy()
    signals = primary_signals.to_numpy()
    n_samples = len(df)

    meta_labels = np.full(n_samples, np.nan)
    ret_at_touch = np.full(n_samples, np.nan)
    touch_indices = np.full(n_samples, np.nan)

    for i in range(n_samples):
        # Skip if no primary conviction signal
        if signals[i] == 0:
            continue

        p_entry = prices[i]
        vol = volatility[i]
        sig = signals[i]

        if np.isnan(vol) or vol <= 0:
            continue

        # Compute dynamic barriers scaled by Parkinson volatility[cite: 1]
        barrier_width = max(pt_sl_ratio * vol, min_ret)
        upper_barrier = np.log(1.0 + barrier_width)
        lower_barrier = -np.log(1.0 + barrier_width)

        # Vertical barrier horizon index[cite: 1]
        t_max = min(i + horizon, n_samples - 1)
        
        # Path-dependent evaluation
        event_label = 0
        terminal_ret = 0.0
        tau_idx = t_max

        for j in range(i + 1, t_max + 1):
            cumulative_log_ret = np.log(prices[j] / p_entry)
            directional_ret = cumulative_log_ret * sig

            # Upper Barrier Hit (Success)[cite: 1]
            if directional_ret >= upper_barrier:
                event_label = 1
                terminal_ret = directional_ret
                tau_idx = j
                break

            # Lower Barrier Hit (Stop-Loss)[cite: 1]
            elif directional_ret <= lower_barrier:
                event_label = 0
                terminal_ret = directional_ret
                tau_idx = j
                break

        # If vertical barrier reached without boundary breach[cite: 1]
        if tau_idx == t_max and event_label == 0:
            terminal_ret = np.log(prices[t_max] / p_entry) * sig
            event_label = 1 if terminal_ret > 0 else 0

        meta_labels[i] = event_label
        ret_at_touch[i] = terminal_ret
        touch_indices[i] = tau_idx

    result_df = pd.DataFrame(
        {
            "primary_signal": signals,
            "meta_label": meta_labels,
            "return_at_touch": ret_at_touch,
            "touch_index": touch_indices,
        },
        index=df.index,
    )
    return result_df.dropna(subset=["meta_label"])