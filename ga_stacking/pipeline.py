"""
Main script: orchestrate GA-stacking end-to-end.
"""
import pandas as pd
import numpy as np
import time
import json
from typing import Dict, Tuple, List, Any, Optional

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature import generate_meta_features
from ga import GA_weighted
from utils import split_and_scale, train_base_models, ensemble_predict, evaluate_metrics
import config
from base_models import BASE_MODELS, META_MODELS
from sklearn.metrics import precision_score
from joblib import parallel_backend



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
        """Train the GA-Stacking ensemble."""
        start_time = time.time()
        
        # Step 1: Generate meta-features using cross-validation
        if self.verbose:
            print("Generating meta-features...")
        meta_X_train = generate_meta_features(
            X_train, y_train, self.base_models, n_splits=self.cv_folds, n_jobs=-1
        )        
        # Step 2: Train base models on full training data
        if self.verbose:
            print("Training base models...")
        with parallel_backend("loky"):
            self.trained_base_models = train_base_models(X_train, y_train, self.base_models, n_jobs=-1)
        
        # Step 3: Generate meta-features for validation set
        if self.verbose:
            print("Generating validation meta-features...")
        meta_X_val = np.column_stack([
            model.predict_proba(X_val)[:, 1]
            for model in self.trained_base_models.values()
        ])
        
        # Step 4: Run GA to get optimal weights
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
        
        # Step 5: Calculate ensemble metrics
        ens_val_preds = ensemble_predict(meta_X_val, self.best_weights)
        val_metrics = evaluate_metrics(y_val, ens_val_preds)
        
        training_time = time.time() - start_time
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity(X_val, y_val)
        
        # Calculate generalization score
        gen_score = self._calculate_generalization(X_train, y_train, X_val, y_val)
        
        # Calculate GA convergence rate
        if len(self.convergence_history) > 1:
            initial = self.convergence_history[0]
            final = self.convergence_history[-1]
            total_improvement = final - initial
            
            # Find where we reached 90% of improvement
            target = initial + 0.9 * total_improvement
            for i, score in enumerate(self.convergence_history):
                if score >= target:
                    convergence_rate = 1.0 - (i / len(self.convergence_history))
                    break
            else:
                convergence_rate = 0.0
        else:
            convergence_rate = 0.5
        
        results = {
            "val_metrics": val_metrics,
            "best_weights": self.best_weights.tolist(),
            "model_names": self.model_names,
            "training_time": training_time,
            "diversity_score": diversity_score,
            "generalization_score": gen_score,
            "convergence_rate": convergence_rate,
            "convergence_history": self.convergence_history
        }
        
        return results
    
    def predict(self, X):
        """Make predictions using the trained ensemble."""
        if self.trained_base_models is None or self.best_weights is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Generate predictions from each base model
        meta_X = np.column_stack([
            model.predict_proba(X)[:, 1] 
            for model in self.trained_base_models.values()
        ])
        
        # Generate ensemble prediction
        return ensemble_predict(meta_X, self.best_weights)
    
    def get_ensemble_state(self):
        """Get the trained ensemble state for serialization."""
        if self.trained_base_models is None or self.best_weights is None:
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
            
            # Add model coefficients if available
            if hasattr(model, 'coef_'):
                params["coef"] = model.coef_.tolist()
            if hasattr(model, 'intercept_'):
                params["intercept"] = model.intercept_.tolist()
            
            model_parameters.append(params)
        
        # Create ensemble state
        ensemble_state = {
            "model_parameters": model_parameters,
            "weights": self.best_weights.tolist(),
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
        meta_X_train = np.column_stack([
            model.predict_proba(X_train)[:, 1]
            for model in self.trained_base_models.values()
        ])
        
        meta_X_val = np.column_stack([
            model.predict_proba(X_val)[:, 1]
            for model in self.trained_base_models.values()
        ])
        
        # Make ensemble predictions
        train_preds = ensemble_predict(meta_X_train, self.best_weights)
        val_preds = ensemble_predict(meta_X_val, self.best_weights)
        
        # Calculate Precision
        train_precision = precision_score(y_train, train_preds > 0.5, pos_label=1)
        val_precision = precision_score(y_val, val_preds > 0.5, pos_label=1)
        
        # Calculate generalization score
        # Lower difference between train and val is better
        diff = abs(train_precision - val_precision)
        
        # Return a score between 0 and 1 where higher is better
        return max(0, 1.0 - diff)

    def save_base_models(self, save_path):
        """Save trained base models to disk for future rounds.
        
        Parameters:
        -----------
        save_path : Path to save models
        """
        if self.trained_base_models is None:
            raise ValueError("No trained base models to save")
        
        import joblib
        import os
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save each model
        for name, model in self.trained_base_models.items():
            model_path = os.path.join(save_path, f"{name}.joblib")
            joblib.dump(model, model_path)
        
        # Save metadata
        model_names = list(self.trained_base_models.keys())
        metadata = {
            "model_names": model_names,
            "timestamp": time.time(),
            "weights": self.best_weights.tolist() if self.best_weights is not None else None
        }
        
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"Saved {len(model_names)} base models to {save_path}")

    def load_base_models(self, load_path):
        """Load base models from disk.
        
        Parameters:
        -----------
        load_path : Path where models are stored
        
        Returns:
        --------
        Dictionary of loaded base models
        """
        import joblib
        import os
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model path {load_path} not found")
        
        # Load metadata to get model names
        with open(os.path.join(load_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        model_names = metadata.get("model_names", [])
        
        # Load each model
        loaded_models = {}
        for name in model_names:
            model_path = os.path.join(load_path, f"{name}.joblib")
            if os.path.exists(model_path):
                loaded_models[name] = joblib.load(model_path)
        
        if self.verbose:
            print(f"Loaded {len(loaded_models)} base models from {load_path}")
        
        # Optionally restore weights
        if "weights" in metadata and metadata["weights"] is not None:
            self.best_weights = np.array(metadata["weights"])
        
        self.trained_base_models = loaded_models
        return loaded_models
    
    def fine_tune(self, X_train, y_train, X_val, y_val, base_models=None, weights=None):
        """Fine-tune pre-trained base models instead of training from scratch.
        
        Parameters:
        -----------
        X_train : Training features
        y_train : Training labels
        X_val : Validation features
        y_val : Validation labels
        base_models : Dictionary of pre-trained base models to fine-tune
        weights : Optional pre-defined weights for the ensemble
        
        Returns:
        --------
        Dictionary with fine-tuning results
        """
        start_time = time.time()
        
        # Use provided base models or fail if none provided
        if base_models is None:
            raise ValueError("Base models must be provided for fine-tuning")
        
        self.trained_base_models = base_models
        
        # Step 1: Fine-tune each base model with fewer iterations/epochs
        if self.verbose:
            print("Fine-tuning base models...")
        
        for model_name, model in self.trained_base_models.items():
            # Fine-tune based on model type
            if hasattr(model, 'warm_start'):
                # For models supporting warm start
                original_warm_start = model.warm_start
                model.warm_start = True
                model.fit(X_train, y_train)
                model.warm_start = original_warm_start
            elif hasattr(model, 'partial_fit') and callable(getattr(model, 'partial_fit')):
                # For models supporting incremental learning
                model.partial_fit(X_train, y_train)
            else:
                # Default: just refit with current weights as starting point
                model.fit(X_train, y_train)
        
        # Step 2: Generate meta-features for validation set
        if self.verbose:
            print("Generating validation meta-features...")
        meta_X_val = np.column_stack([
            model.predict_proba(X_val)[:, 1]
            for model in self.trained_base_models.values()
        ])
        
        # Step 3: Use provided weights or run lighter GA
        if weights is not None:
            self.best_weights = np.array(weights)
            self.convergence_history = []  # No convergence history when using provided weights
        else:
            # Generate meta-features for training set
            if self.verbose:
                print("Generating meta-features for GA optimization...")
            meta_X_train = generate_meta_features(
                X_train, y_train, {k: self.trained_base_models[k] for k in self.model_names}, 
                n_splits=self.cv_folds, n_jobs=-1
            )
            
            # Run GA with fewer generations
            if self.verbose:
                print("Running lightweight GA optimization...")
            reduced_generations = max(5, self.generations // 5)  # Use fewer generations
            self.best_weights, convergence = GA_weighted(
                meta_X_train, y_train, meta_X_val, y_val,
                pop_size=self.pop_size,
                generations=reduced_generations,
                crossover_prob=self.crossover_prob,
                mutation_prob=self.mutation_prob,
                metric=self.metric,
                verbose=self.verbose
            )
            self.convergence_history = convergence
        
        # Step 4: Calculate ensemble metrics
        ens_val_preds = ensemble_predict(meta_X_val, self.best_weights)
        val_metrics = evaluate_metrics(y_val, ens_val_preds)
        
        training_time = time.time() - start_time
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity(X_val, y_val)
        
        # Calculate generalization score
        gen_score = self._calculate_generalization(X_train, y_train, X_val, y_val)
        
        # Calculate GA convergence rate or use default if no GA was run
        if len(self.convergence_history) > 1:
            initial = self.convergence_history[0]
            final = self.convergence_history[-1]
            total_improvement = final - initial
            
            # Find where we reached 90% of improvement
            target = initial + 0.9 * total_improvement
            for i, score in enumerate(self.convergence_history):
                if score >= target:
                    convergence_rate = 1.0 - (i / len(self.convergence_history))
                    break
            else:
                convergence_rate = 0.0
        else:
            convergence_rate = 0.75  # Higher default for fine-tuning
        
        results = {
            "val_metrics": val_metrics,
            "best_weights": self.best_weights.tolist(),
            "model_names": self.model_names,
            "training_time": training_time,
            "diversity_score": diversity_score,
            "generalization_score": gen_score,
            "convergence_rate": convergence_rate,
            "convergence_history": self.convergence_history
        }
        
        return results

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
    print(f"PRECISION: {test_metrics['precision']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    
    # Save ensemble state
    ensemble_state = pipeline.get_ensemble_state()
    with open("ensemble_state.json", "w") as f:
        json.dump(ensemble_state, f, indent=2)
    
    print("\nEnsemble state saved to ensemble_state.json")