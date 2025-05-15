"""
Module: utils
-------------
General helper functions: data splitting, model training, predictions, metrics.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report
)
from sklearn.base import clone

from base_models import BASE_MODELS, META_MODELS


def split_and_scale(data, target_col='Class', test_size=0.3, random_state=42):
    """
    Split raw DataFrame into train/val/test, scale and balance.

    Returns:
      X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler
    """
    X = data.drop(['Time', target_col], axis=1)
    y = data[target_col]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2/3, stratify=y_temp, random_state=random_state)
    scaler = RobustScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    X_train_s, y_train = SMOTE().fit_resample(X_train_s, y_train)
    return X_train_s, X_val_s, X_test_s, y_train.values, y_val.values, y_test.values, scaler


def train_base_models(X, y, model_dict=BASE_MODELS):
    """
    Fit each base model on (X, y).
    Returns list of fitted models.
    """
    models = {}
    for name, m in model_dict.items():
        m_clone = clone(m)
        m_clone.fit(X, y)
        models[name] = m_clone
    return models


def ensemble_predict(meta_X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute weighted ensemble probabilities.
    """
    return np.dot(meta_X, weights)


def evaluate_metrics(y_true, y_proba, threshold=0.5):
    """
    Calculate various metrics for model evaluation.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Handle input validation
    if isinstance(y_proba, list):
        y_proba = np.array(y_proba)
    
    # Convert probabilities to binary predictions
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics with error handling
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = 0.5  # Default AUC value for failure cases
    
    try:
        f1 = f1_score(y_true, y_pred)
    except Exception:
        f1 = 0.0
    
    try:
        accuracy = accuracy_score(y_true, y_pred)
    except Exception:
        accuracy = 0.0
    
    try:
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        if '1' in report_dict:
            precision = report_dict['1']['precision']
            recall = report_dict['1']['recall']
        else:
            # Handle case where class '1' doesn't exist in the results
            precision = 0.0
            recall = 0.0
    except Exception as e:
        precision = 0.0
        recall = 0.0
    
    # Print metrics for debugging
    print('AUC :', auc)
    print('F1  :', f1)
    
    try:
        # Only print classification report if it works
        print(classification_report(y_true, y_pred))
    except Exception:
        # If it fails, print basic metrics instead
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # Create metrics dictionary
    metrics = {
        'auc': float(auc),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
    }
    
    return metrics