"""
Module: feature
---------------
Functions for generating meta-features via K-Fold stacking
and meta-predictions from trained base models.
"""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

def generate_meta_features(
    X: np.ndarray,
    y: np.ndarray,
    model_dict: dict,
    n_splits: int = 5
) -> np.ndarray:
    """
    Generate training meta-features using K-Fold cross-validation.

    Args:
        X         : array-like, shape (n_samples, n_features)
        y         : array-like, shape (n_samples,)
        model_dict: dict of name->estimator
        n_splits  : int, number of CV folds

    Returns:
        meta_X: np.ndarray of shape (n_samples, n_models)
                each column is predicted probability of class 1
                from a base model on held-out folds.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = X.shape[0]
    n_models = len(model_dict)
    meta_X = np.zeros((n_samples, n_models))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    keys = list(model_dict.keys())

    for idx, name in enumerate(keys):
        base_model = model_dict[name]
        for train_idx, val_idx in skf.split(X, y):
            m = clone(base_model)
            m.fit(X[train_idx], y[train_idx])
            meta_X[val_idx, idx] = m.predict_proba(X[val_idx])[:, 1]
    return meta_X