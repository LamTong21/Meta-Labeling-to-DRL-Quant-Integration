from typing import Any, Dict, Literal, Optional, Tuple, Union
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

from src.labeling.calibrator import ProbabilityCalibrator


class MetaLabelingEnsemble:
    """Supervised meta-labeling classification engine for Modeling09.
    
    Combines tree-based gradient boosting estimators (XGBoost, LightGBM, Random Forest)
    to predict informational event success: P(y_t* = 1 | X_t), followed by non-parametric
    probability calibration for bet sizing.
    """
    def __init__(
        self,
        estimator_type: Literal["xgboost", "lightgbm", "random_forest"] = "lightgbm",
        hyperparams: Optional[Dict[str, Any]] = None,
        calibration_method: Literal["isotonic", "sigmoid"] = "isotonic",
        random_state: int = 42,
    ):
        self.estimator_type = estimator_type
        self.hyperparams = hyperparams or {}
        self.calibration_method = calibration_method
        self.random_state = random_state

        self.model = self._initialize_model()
        self.calibrator = ProbabilityCalibrator(method=self.calibration_method)
        self.is_fitted = False

    def _initialize_model(self):
        """Builds the primary binary classification estimator."""
        if self.estimator_type == "lightgbm":
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 5,
                "num_leaves": 31,
                "random_state": self.random_state,
                "verbose": -1,
            }
            default_params.update(self.hyperparams)
            return lgb.LGBMClassifier(**default_params)

        elif self.estimator_type == "xgboost":
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 5,
                "random_state": self.random_state,
                "eval_metric": "logloss",
            }
            default_params.update(self.hyperparams)
            return xgb.XGBClassifier(**default_params)

        elif self.estimator_type == "random_forest":
            default_params = {
                "n_estimators": 150,
                "max_depth": 6,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            default_params.update(self.hyperparams)
            return RandomForestClassifier(**default_params)

        else:
            raise ValueError(f"Unsupported estimator type: {self.estimator_type}")

    def fit_with_cv_calibration(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        n_splits: int = 5,
    ) -> Dict[str, float]:
        """Trains the model across chronological TimeSeriesSplit folds to collect
        out-of-fold probability predictions, avoiding lookahead bias during calibration.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_probs = np.zeros(len(X))
        oof_targets = np.zeros(len(X))
        eval_indices = []

        X_mat = X.to_numpy()
        y_vec = y.to_numpy() if isinstance(y, pd.Series) else y

        for train_idx, val_idx in tscv.split(X_mat):
            X_train, X_val = X_mat[train_idx], X_mat[val_idx]
            y_train, y_val = y_vec[train_idx], y_vec[val_idx]

            fold_model = self._initialize_model()
            fold_model.fit(X_train, y_train)

            val_preds = fold_model.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = val_preds
            oof_targets[val_idx] = y_val
            eval_indices.extend(val_idx)

        # Fit probability calibrator strictly on out-of-fold cross-validation estimates
        eval_indices = np.array(eval_indices)
        self.calibrator.fit(oof_probs[eval_indices], oof_targets[eval_indices])

        # Train final model on the full historical dataset
        self.model.fit(X_mat, y_vec)
        self.is_fitted = True

        # Validation diagnostics
        cv_auc = roc_auc_score(oof_targets[eval_indices], oof_probs[eval_indices])
        cv_logloss = log_loss(oof_targets[eval_indices], oof_probs[eval_indices])
        cv_brier = brier_score_loss(oof_targets[eval_indices], oof_probs[eval_indices])

        return {
            "cv_roc_auc": float(cv_auc),
            "cv_log_loss": float(cv_logloss),
            "cv_brier_score": float(cv_brier),
        }

    def predict_conviction(
        self, 
        X: pd.DataFrame, 
        primary_signals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Infers calibrated event probabilities and sizes active bets:
            w_t = sign(y_hat_t) * 2 * (P_calibrated - 0.5)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit_with_cv_calibration first.")

        X_mat = X.to_numpy()
        raw_probs = self.model.predict_proba(X_mat)[:, 1]
        calibrated_probs = self.calibrator.predict_proba(raw_probs)
        bet_sizes = self.calibrator.compute_bet_size(calibrated_probs, primary_signals)

        return calibrated_probs, bet_sizes