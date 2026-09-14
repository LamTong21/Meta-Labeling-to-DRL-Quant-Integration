from typing import Literal, Optional
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression[cite: 1, 2]
from sklearn.linear_model import LogisticRegression[cite: 1, 2]


class ProbabilityCalibrator:
    """Non-parametric or parametric probability calibration for GBDT confidence scores[cite: 1, 2].
    
    Transforms uncalibrated raw scores into empirical posterior margins[cite: 1]:
        Isotonic Objective: min_m sum_{i=1}^M (y_i* - m(p_hat_i))^2 subject to m non-decreasing[cite: 1]
        Bet Sizing mapping: w_t = sign(y_hat_t) * max(0, 2 * (p_cal - 0.5))[cite: 1]
    """
    def __init__(self, method: Literal["isotonic", "sigmoid"] = "isotonic"):
        self.method = method
        self.calibrator = None

    def fit(self, y_raw_probs: np.ndarray, y_true: np.ndarray):
        """Fits calibration curve on out-of-fold validation predictions."""
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(y_raw_probs, y_true)
        elif self.method == "sigmoid":
            # Platt Sigmoid Scaling via univariate Logistic Regression[cite: 1]
            self.calibrator = LogisticRegression(C=1e5, solver="lbfgs")
            self.calibrator.fit(y_raw_probs.reshape(-1, 1), y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        return self

    def predict_proba(self, y_raw_probs: np.ndarray) -> np.ndarray:
        """Transforms raw model output into calibrated probabilities: P_calibrated in [0, 1][cite: 1]."""
        if self.calibrator is None:
            raise ValueError("Calibrator is not fitted yet.")

        if self.method == "isotonic":
            return self.calibrator.predict(y_raw_probs)
        elif self.method == "sigmoid":
            return self.calibrator.predict_proba(y_raw_probs.reshape(-1, 1))[:, 1]

    @staticmethod
    def compute_bet_size(
        calibrated_probs: np.ndarray, 
        primary_signals: np.ndarray, 
        threshold: float = 0.5
    ) -> np.ndarray:
        """Computes continuous conviction bet weights:
            w_t = sign(y_hat_t) * g(p_cal, t), where g(p) = max(0, 2*(p - 0.5))[cite: 1]
        """
        raw_margin = np.maximum(0.0, 2.0 * (calibrated_probs - threshold))[cite: 1]
        return np.sign(primary_signals) * raw_margin[cite: 1]