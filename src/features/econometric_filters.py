from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch[cite: 1, 2]
from statsmodels.tsa.stattools import adfuller, kpss[cite: 1, 2]


def check_stationarity(
    series: pd.Series, 
    significance_level: float = 0.05
) -> Dict[str, Union[float, bool, str]]:
    """Evaluates weak stationarity using joint Augmented Dickey-Fuller (ADF)
    and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test rejection bounds.

    Mathematical formulation:
        ADF:  Delta X_t = alpha + beta*t + gamma*X_{t-1} + sum(delta_p*Delta X_{t-p}) + eps_t
              H0: gamma = 0 (Unit Root present -> Non-stationary)
        KPSS: H0: Stationary around a deterministic trend/level.
    """
    clean_series = series.dropna()
    if len(clean_series) < 20:
        return {
            "adf_pvalue": np.nan,
            "kpss_pvalue": np.nan,
            "is_stationary": False,
            "conclusion": "Insufficient sample size",
        }

    # 1. ADF Test (H0: Non-stationary)
    adf_res = adfuller(clean_series, autolag="AIC")[cite: 1, 2]
    adf_pvalue = float(adf_res[1])

    # 2. KPSS Test (H0: Stationary)
    kpss_res = kpss(clean_series, regression="c", nlags="auto")[cite: 1, 2]
    kpss_pvalue = float(kpss_res[1])

    # Case A: Reject ADF H0 (p < sig) and Fail to reject KPSS H0 (p >= sig) -> Strictly Stationary
    # Case B: Reject ADF H0 and Reject KPSS H0 -> Difference/Trend Stationary
    is_strictly_stationary = (adf_pvalue < significance_level) and (kpss_pvalue >= significance_level)
    is_weakly_stationary = adf_pvalue < significance_level

    conclusion = "Strictly Stationary" if is_strictly_stationary else (
        "Weakly Stationary" if is_weakly_stationary else "Non-Stationary"
    )

    return {
        "adf_pvalue": adf_pvalue,
        "kpss_pvalue": kpss_pvalue,
        "is_stationary": is_weakly_stationary,
        "conclusion": conclusion,
    }


def test_autocorrelation_and_arch(
    residuals: pd.Series, 
    lags: int = 10, 
    significance_level: float = 0.05
) -> Dict[str, Union[float, bool]]:
    """Evaluates serial correlation via Ljung-Box test and conditional heteroskedasticity 
    via Engle's ARCH-LM test on residuals.

    Mathematical formulation:
        Ljung-Box: Q_{LB} = T(T+2) * sum_{k=1}^m (rho_hat_k^2 / (T-k)) ~ chi^2(m)
        ARCH-LM:   eps_hat_t^2 = alpha_0 + sum_{i=1}^q (alpha_i * eps_hat_{t-i}^2) + nu_t
                   H0: alpha_1 = ... = alpha_q = 0 (No ARCH effects)
    """
    clean_res = residuals.dropna()
    if len(clean_res) <= lags:
        return {
            "lb_pvalue": np.nan,
            "has_autocorrelation": False,
            "arch_pvalue": np.nan,
            "has_arch_effect": False,
        }

    # Ljung-Box test for serial correlation
    lb_df = acorr_ljungbox(clean_res, lags=[lags], return_clean=True)[cite: 1, 2]
    lb_pvalue = float(lb_df["lb_pvalue"].iloc[-1])

    # Engle's ARCH-LM test for volatility clustering
    arch_res = het_arch(clean_res, maxlag=lags)[cite: 1, 2]
    arch_pvalue = float(arch_res[1])  # p-value of LM test

    return {
        "lb_pvalue": lb_pvalue,
        "has_autocorrelation": lb_pvalue < significance_level,
        "arch_pvalue": arch_pvalue,
        "has_arch_effect": arch_pvalue < significance_level,
    }


class EconometricFilterEngine:
    """Pre-filtering pipeline to ensure candidate features are stationary 
    and statistically well-behaved prior to boosting ingest.
    """
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def filter_features(self, df_features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, dict]]:
        """Filters out non-stationary columns across a multivariate dataframe."""
        retained_features = []
        audit_report = {}

        for col in df_features.columns:
            series = df_features[col]
            stat_audit = check_stationarity(series, self.significance_level)
            audit_report[col] = stat_audit

            if stat_audit["is_stationary"]:
                retained_features.append(col)

        return df_features[retained_features].copy(), audit_report