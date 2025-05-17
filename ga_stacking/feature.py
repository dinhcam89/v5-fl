import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

def generate_meta_features(
    X: np.ndarray,
    y: np.ndarray,
    model_dict: dict,
    n_splits: int = 5,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Generate training meta-features using K-Fold cross-validation.
    Parallelized across models and folds using joblib.

    Args:
        X         : array-like, shape (n_samples, n_features)
        y         : array-like, shape (n_samples,)
        model_dict: dict of name->estimator
        n_splits  : int, number of CV folds
        n_jobs    : number of parallel jobs (default: -1 for all cores)

    Returns:
        meta_X: np.ndarray of shape (n_samples, n_models)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = X.shape[0]
    model_names = list(model_dict.keys())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Prepare jobs: one job per (model, fold)
    jobs = []
    for model_idx, name in enumerate(model_names):
        base_model = model_dict[name]
        for train_idx, val_idx in skf.split(X, y):
            jobs.append((model_idx, name, clone(base_model), train_idx, val_idx))

    # Define the worker function
    def fit_model(model_idx, name, model, train_idx, val_idx):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict_proba(X[val_idx])[:, 1]
        return model_idx, val_idx, preds

    # Run jobs in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_model)(*job) for job in jobs
    )

    # Combine results into meta_X
    meta_X = np.zeros((n_samples, len(model_dict)))
    for model_idx, val_idx, preds in results:
        meta_X[val_idx, model_idx] = preds

    return meta_X

def predict_meta_features(
    model_dict: dict,
    X: np.ndarray,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Generate meta-features for test data using trained base models.
    
    Args:
        model_dict: dict of name->trained estimator
        X         : array-like, shape (n_samples, n_features)
        n_jobs    : number of parallel jobs (default: -1 for all cores)
        
    Returns:
        meta_X: np.ndarray of shape (n_samples, n_models)
    """
    X = np.asarray(X)
    n_samples = X.shape[0]
    model_names = list(model_dict.keys())
    
    # Prepare jobs: one job per model
    jobs = []
    for model_idx, name in enumerate(model_names):
        # Use the already fitted model, don't clone it
        base_model = model_dict[name]
        jobs.append((model_idx, name, base_model, X))
    
    # Define the worker function
    def predict_model(model_idx, name, model, X):
        try:
            preds = model.predict_proba(X)[:, 1]
        except (AttributeError, IndexError) as e:
            # Fallback if predict_proba isn't available
            print(f"Warning: {name} doesn't support predict_proba: {str(e)}")
            preds = model.predict(X).astype(float)
        return model_idx, preds
    
    # Run jobs in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(predict_model)(*job) for job in jobs
    )
    
    # Combine results into meta_X
    meta_X = np.zeros((n_samples, len(model_dict)))
    for model_idx, preds in results:
        meta_X[:, model_idx] = preds
    
    return meta_X