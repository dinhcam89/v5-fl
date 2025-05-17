"""
Main script: orchestrate GA-stacking end-to-end with meta-learner integration.
"""
import pandas as pd
import numpy as np
import time
import json
from typing import Dict, Tuple, List, Any, Optional
from sklearn.base import clone

import copy

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature import generate_meta_features, predict_meta_features
from ga import GA_weighted
from utils import split_and_scale, train_base_models, evaluate_metrics
from base_models import BASE_MODELS, META_MODELS
from sklearn.metrics import f1_score
import config


class GAStackingPipeline:
    def __init__(
        self,
        base_models=BASE_MODELS,
        meta_models=META_MODELS,
        pop_size=config.POP_SIZE,
        generations=config.GENERATIONS,
        cv_folds=config.CV_FOLDS,
        crossover_prob=config.CROSSOVER_PROB,
        mutation_prob=config.MUTATION_PROB,
        metric=config.METRIC,
        verbose=True
    ):
        self.base_models = base_models
        self.meta_models = meta_models
        self.pop_size = pop_size
        self.generations = generations
        self.cv_folds = cv_folds
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.metric = metric
        self.verbose = verbose
        
        # Initialized during training
        self.trained_base_models = None
        self.best_weights = None
        self.meta_model = None
        self.convergence_history = []
        self.model_names = list(base_models.keys())
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the GA-Stacking ensemble with a meta-learner."""
        start_time = time.time()
        
        # Step 1: Train base models
        if self.verbose:
            print("Training base models...")
        self.trained_base_models = train_base_models(X_train, y_train, self.base_models)
        
        # Step 2: Generate meta-features
        if self.verbose:    
            print("Generating meta-features...")
        meta_X_train = generate_meta_features(X_train, y_train, self.trained_base_models, n_splits=self.cv_folds)
        meta_X_val = predict_meta_features(self.trained_base_models, X_val)
        
        # Step 3: Run GA to get optimal weights
        if self.verbose:
            print("Running GA optimization...")
        self.best_weights, convergence = GA_weighted(
            meta_X_train, y_train, meta_X_val, y_val,
            pop_size=self.pop_size,
            generations=self.generations,
            crossover_prob=self.crossover_prob,
            mutation_prob=self.mutation_prob,
            metric=self.metric,
            verbose=self.verbose
        )
        self.convergence_history = convergence
        
        # Step 4: Train the meta-learner
        if self.verbose:
            print("Training meta-learner...")
            
        # Make sure we have proper dimensions for the meta-learner input
        # This is a key fix - ensuring proper input shape for the meta-learner
        meta_level_val = meta_X_val.dot(self.best_weights)
        
        # If the meta-learner expects a 2D array, reshape accordingly
        if len(meta_level_val.shape) == 1:
            meta_level_val = meta_level_val.reshape(-1, 1)
            
        # Ensure y_val is the correct shape too
        y_val_reshaped = y_val.copy()
        if len(y_val_reshaped.shape) == 2 and y_val_reshaped.shape[1] == 1:
            y_val_reshaped = y_val_reshaped.ravel()
            
        # Create and train the meta-learner
        self.meta_model = copy.deepcopy(self.meta_models["meta_learner"])
        
        
        # Add error checking when fitting meta-learner
        try:
            self.meta_model.fit(meta_level_val, y_val_reshaped)
        except Exception as e:
            print(f"Error fitting meta-learner: {str(e)}")
            print(f"Meta features shape: {meta_level_val.shape}, Target shape: {y_val_reshaped.shape}")
            print(f"Meta features sample: {meta_level_val[:5]}")
            print(f"Target sample: {y_val_reshaped[:5]}")
            raise  # Re-raise the exception after logging details
        
        # Step 5: Evaluate the meta-learner on the validation set
        try:
            meta_val_preds = self.meta_model.predict_proba(meta_level_val)[:, 1]
        except Exception as e:
            print(f"Error in predict_proba: {str(e)}")
            # Some models might not have predict_proba
            meta_val_preds = (self.meta_model.predict(meta_level_val) > 0.5).astype(float)
        
        val_metrics = evaluate_metrics(y_val, meta_val_preds)
        
        training_time = time.time() - start_time
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity(X_val, y_val)
        
        # Calculate generalization score
        gen_score = self._calculate_generalization(X_train, y_train, X_val, y_val)
        
         # Calculate convergence rate (new code)
        if len(self.convergence_history) > 1:
            # Convergence rate is how quickly the GA optimization converged
            # A simple measure: (final_score - initial_score) / initial_score
            initial_score = self.convergence_history[0]
            final_score = self.convergence_history[-1]
            
            # Avoid division by zero
            if initial_score > 0:
                convergence_rate = min(1.0, (final_score - initial_score) / initial_score)
            else:
                convergence_rate = 0.5  # Default value if initial score is 0
        else:
            convergence_rate = 0.5  # Default value if not enough generations
        
        results = {
            "val_metrics": val_metrics,
            "best_weights": self.best_weights.tolist(),
            "model_names": self.model_names,
            "training_time": training_time,
            "diversity_score": diversity_score,
            "generalization_score": gen_score,
            "convergence_history": self.convergence_history,
            "convergence_rate": convergence_rate  # Add this line
        }
        
        return results
    
    def predict(self, X):
        """Make predictions using the trained ensemble with a meta-learner."""
        if self.trained_base_models is None or self.best_weights is None or self.meta_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Generate meta-features from base models
            meta_X = predict_meta_features(self.trained_base_models, X)
            
            # Generate meta-level features using GA weights
            meta_level_X = meta_X.dot(self.best_weights)
            
            # Reshape if needed
            if len(meta_level_X.shape) == 1:
                meta_level_X = meta_level_X.reshape(-1, 1)
            
            # Use the meta-learner to make predictions
            try:
                # Try using predict_proba first (for classifiers)
                return self.meta_model.predict_proba(meta_level_X)[:, 1]
            except (AttributeError, IndexError):
                # Fall back to predict for regressors or if predict_proba fails
                return self.meta_model.predict(meta_level_X)
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            # Return default predictions (all zeros or 0.5 for binary classification)
            return np.zeros(len(X)) if self.metric == 'f1' else np.full(len(X), 0.5)
    
    def get_ensemble_state(self):
        """Get the trained ensemble state for serialization."""
        if self.trained_base_models is None or self.best_weights is None or self.meta_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get model parameters for each base model
        model_parameters = []
        for name, model in self.trained_base_models.items():
            # Extract model parameters (this is simplified and not complete)
            params = {
                "estimator": name,
                "model_type": name,
            }
            
            # Add model-specific parameters
            if hasattr(model, 'get_params'):
                model_params = model.get_params()
                params.update(model_params)
            
            model_parameters.append(params)
        
        # Get meta-learner parameters
        meta_model_params = {
            "meta_model_type": type(self.meta_model).__name__,
            "meta_model_params": self.meta_model.get_params() if hasattr(self.meta_model, 'get_params') else {}
        }
        
        # Create ensemble state
        ensemble_state = {
            "model_parameters": model_parameters,
            "weights": self.best_weights.tolist(),
            "meta_model": meta_model_params,
            "model_names": self.model_names
        }
        
        return ensemble_state
    
    def _calculate_diversity(self, X, y):
        """Calculate diversity among base models."""
        if self.trained_base_models is None:
            return 0.0
        
        # Get predictions from all models
        all_preds = []
        for model in self.trained_base_models.values():
            # Binarize predictions
            preds = (model.predict_proba(X)[:, 1] > 0.5).astype(int)
            all_preds.append(preds)
        
        # Calculate pairwise disagreement
        n_models = len(all_preds)
        if n_models < 2:
            return 0.0
        
        disagreement_sum = 0
        comparison_count = 0
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                # Count how often models i and j make different predictions
                disagreement = np.mean(all_preds[i] != all_preds[j])
                disagreement_sum += disagreement
                comparison_count += 1
        
        # Normalize by number of comparisons
        return disagreement_sum / comparison_count if comparison_count > 0 else 0.0
    
    def _calculate_generalization(self, X_train, y_train, X_val, y_val):
        """Calculate generalization score (difference between train and val performance)."""
        if self.trained_base_models is None or self.best_weights is None:
            return 0.0
        
        # Generate meta-features
        meta_X_train = predict_meta_features(self.trained_base_models, X_train)
        meta_X_val = predict_meta_features(self.trained_base_models, X_val)
        
        # Make ensemble predictions
        train_preds = meta_X_train.dot(self.best_weights)
        val_preds = meta_X_val.dot(self.best_weights)
        
        # Calculate AUC
        train_f1 = f1_score(y_train, train_preds >0.5, pos_label=1)
        val_f1 = f1_score(y_val, val_preds >0.5, pos_label=1)
        
        # Calculate generalization score
        # Lower difference between train and val is better
        diff = abs(train_f1 - val_f1)
        
        # Return a score between 0 and 1 where higher is better
        return max(0, 1.0 - diff)


if __name__ == '__main__':
    # Demo usage
    import pandas as pd
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                               n_redundant=5, n_classes=2, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # Create and train GA-Stacking pipeline
    pipeline = GAStackingPipeline(
        generations=10,  # Fewer generations for demo
        pop_size=20,     # Smaller population for demo
        verbose=True
    )
    
    results = pipeline.train(X_train, y_train, X_val, y_val)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    test_metrics = evaluate_metrics(y_test, y_pred)
    print("\nTest set metrics:")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    
    # Save ensemble state
    ensemble_state = pipeline.get_ensemble_state()
    with open("ensemble_state.json", "w") as f:
        json.dump(ensemble_state, f, indent=2)
    
    print("\nEnsemble state saved to ensemble_state.json")