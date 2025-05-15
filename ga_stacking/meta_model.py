"""
------------------
Defines the meta-learner for 2-tier stacking.
By default using LogisticRegression but can be replaced.
"""
from sklearn.linear_model import LogisticRegression

# Uninitialized meta-learner
META_MODEL = LogisticRegression(C=0.1, max_iter=200)