from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture[cite: 1, 2]
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression[cite: 1, 2]


class GaussianMixtureRegimeDetector:
    """Detects multi-modal financial return regimes (e.g., Bull, Bear, Sideway)
    using unsupervised Gaussian Mixture Models (GMM).
    """
    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model: Optional[GaussianMixture] = None

    def fit_predict(self, series: pd.Series) -> pd.Series:
        """Fits GMM on log-return or Parkinson volatility series and assigns discrete regime tags."""
        clean_series = series.dropna()
        X = clean_series.to_numpy().reshape(-1, 1)

        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=self.random_state,
            n_init=5,
        )[cite: 1, 2]
        self.model.fit(X)

        regimes = self.model.predict(X)
        return pd.Series(regimes, index=clean_series.index, name="gmm_regime")

    def predict_proba(self, series: pd.Series) -> pd.DataFrame:
        """Outputs posterior probabilities P(Regime = k | X_t)."""
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit_predict first.")

        clean_series = series.dropna()
        X = clean_series.to_numpy().reshape(-1, 1)
        probs = self.model.predict_proba(X)

        return pd.DataFrame(
            probs, 
            index=clean_series.index, 
            columns=[f"regime_prob_{k}" for k in range(self.n_regimes)]
        )


class MarkovSwitchingRegimeDetector:
    """Hamilton (1989) Markov-switching Dynamic Regression to identify regime transitions 
    with state-dependent mean and variance.
    """
    def __init__(self, k_regimes: int = 2, trend: str = "c", switching_variance: bool = True):
        self.k_regimes = k_regimes
        self.trend = trend
        self.switching_variance = switching_variance
        self.fitted_model = None

    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None):
        """Fits the Markov Autoregressive Switching architecture."""
        clean_series = series.dropna()
        clean_exog = exog.loc[clean_series.index] if exog is not None else None

        model = MarkovRegression(
            endog=clean_series,
            k_regimes=self.k_regimes,
            trend=self.trend,
            exog=clean_exog,
            switching_variance=self.switching_variance,
        )[cite: 1, 2]
        
        self.fitted_model = model.fit(disp=False)
        return self

    def get_smoothed_probabilities(self) -> pd.DataFrame:
        """Extracts smoothed regime probabilities P(S_t = j | Omega_T)."""
        if self.fitted_model is None:
            raise ValueError("Model has not been fitted. Call fit first.")
        
        return self.fitted_model.smoothed_marginal_probabilities