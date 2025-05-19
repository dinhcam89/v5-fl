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
from joblib import Parallel, delayed

from base_models import BASE_MODELS, META_MODELS
import logging
logger = logging.getLogger("FL-Client-Ensemble")

def split_and_scale(data, target_col='Class', test_size=0.3, random_state=42):
    """
    A modified version of split_and_scale that does not require a 'Time' column.
    This function handles preprocessing a DataFrame for machine learning.
    
    Args:
        data (pd.DataFrame): Input DataFrame with features and target column
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler
    """
    # Log all columns for debugging
    logger.info(f"DataFrame columns: {data.columns.tolist()}")
    
    # Check if target column exists
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. Available columns: {data.columns.tolist()}")
        
    # Extract target variable
    y = data[target_col]
    
    # Drop 'Time' column if it exists
    if 'Time' in data.columns:
        data = data.drop(['Time'], axis=1)
    
    # Create feature set by dropping the target column
    X = data.drop([target_col], axis=1)
    
    # Log shapes 
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Split into train/temp and temp into val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2/3, stratify=y_temp, random_state=random_state)
    
    # Scale features
    scaler = RobustScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    logger.info(f"Data info before SMOTE:")
    logger.info(f"[BEFORE] Train: {X_train_s.shape}, Validation: {X_val_s.shape}, Test: {X_test_s.shape}")

    # Apply SMOTE to handle class imbalance
    try:
        X_train_s, y_train = SMOTE(random_state=random_state).fit_resample(X_train_s, y_train)
        logger.info(f"SMOTE applied. New training shape: {X_train_s.shape}")
    except Exception as e:
        logger.warning(f"SMOTE failed, using original data: {str(e)}")
    
    # Convert to numpy arrays
    y_train_np = np.array(y_train)
    y_val_np = np.array(y_val)
    y_test_np = np.array(y_test)
    
    logger.info(f"Preprocessing complete.")
    logger.info(f"Train: {X_train_s.shape}, Validation: {X_val_s.shape}, Test: {X_test_s.shape}")
    
    return X_train_s, X_val_s, X_test_s, y_train_np, y_val_np, y_test_np, scaler


def train_base_models(X, y, model_dict=BASE_MODELS, n_jobs: int = -1):
    """
    Huấn luyện song song các base models trên (X, y).
    
    Args:
        X, y: Dữ liệu train
        model_dict: dict tên -> mô hình chưa huấn luyện
        n_jobs: số luồng song song (default: -1 = tất cả CPU cores)
    
    Returns:
        models: dict tên -> mô hình đã huấn luyện
    """

    def fit_model(name, model):
        m_clone = clone(model)
        m_clone.fit(X, y)
        return name, m_clone

    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_model)(name, model) for name, model in model_dict.items()
    )

    return {name: model for name, model in results}


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
    print('F1  :', f1)
    
    try:
        # Only print classification report if it works
        print(classification_report(y_true, y_pred))
    except Exception:
        # If it fails, print basic metrics instead
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # Create metrics dictionary
    metrics = {
        'f1': float(f1),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
    }
    
    return metrics
