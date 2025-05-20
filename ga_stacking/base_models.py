"""
Module: base_models
-------------------
Defines base models for GA-stacking. Add or remove models as needed.
Each model should implement fit() and predict_proba().
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Dictionary mapping model name to uninitialized estimator
BASE_MODELS = {
    'lr':       LogisticRegression(solver='liblinear'),
    'svc':      SVC(probability=True),
    'rf':       RandomForestClassifier(n_estimators=100),
    'knn':      KNeighborsClassifier(n_neighbors=5),
    'catboost': CatBoostClassifier(verbose=0)
    # 'lgbm':     LGBMClassifier(),
    # 'xgb':      XGBClassifier(eval_metric='logloss')
}

# Dictionary for meta-model
META_MODELS = {
    'lg': LogisticRegression(solver='liblinear')
}