from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy import stats[cite: 1, 2]
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression[cite: 1, 2]


def compute_shannon_entropy(series: pd.Series, bins: int = 50) -> float:
    """Computes continuous Shannon Entropy H(X) via empirical histogram probability estimation:
        H(X) = - sum_{x in X} p(x) * ln(p(x))
    """
    clean_series = series.dropna().to_numpy()
    if len(clean_series) == 0:
        return 0.0

    counts, _ = np.histogram(clean_series, bins=bins, density=False)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]  # Mask zero entries to prevent log domain errors

    return float(-np.sum(probs * np.log(probs)))


def filter_by_mutual_information(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    is_classification: bool = True,
    threshold: float = 0.01,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Filters candidate features based on non-linear Mutual Information I(X_j; Y):
        I(X_j; Y) = iint p(x, y) * ln( p(x,y) / (p(x)*p(y)) ) dx dy >= threshold

    Returns:
        Filtered feature matrix and the ranked mutual information score Series.
    """
    X_clean = X.copy()
    
    # Fill remaining NaNs using column median to compute robust nearest neighbor distances
    X_imputed = X_clean.fillna(X_clean.median())

    if is_classification:
        mi_scores = mutual_info_classif(
            X_imputed, 
            y, 
            n_neighbors=n_neighbors, 
            random_state=random_state
        )[cite: 1, 2]
    else:
        mi_scores = mutual_info_regression(
            X_imputed, 
            y, 
            n_neighbors=n_neighbors, 
            random_state=random_state
        )[cite: 1, 2]

    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    selected_features = mi_series[mi_series >= threshold].index.tolist()

    return X[selected_features].copy(), mi_series