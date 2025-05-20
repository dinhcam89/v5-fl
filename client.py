"""
Enhanced Federated Learning Client with GA-Stacking Ensemble optimization.
Extends the base client with local ensemble optimization capabilities.
Uses scikit-learn models exclusively (no PyTorch).
"""

import os
import json
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split

import random
import threading
from web3.exceptions import TransactionNotFound
from hexbytes import HexBytes
import pandas as pd

import faulthandler
faulthandler.enable()

import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

import numpy as np
from sklearn.preprocessing import StandardScaler

from ipfs_connector import IPFSConnector
from blockchain_connector import BlockchainConnector

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import GA-Stacking modules
from ga_stacking.pipeline import GAStackingPipeline
from ga_stacking.base_models import BASE_MODELS, META_MODELS
from ga_stacking.utils import split_and_scale, evaluate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("FL-Client-Ensemble")


class GAStackingClient(fl.client.NumPyClient):
    """Federated Learning client with GA-Stacking ensemble optimization using scikit-learn."""
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        ensemble_size: int = 5,
        ipfs_connector: Optional[IPFSConnector] = None,
        blockchain_connector: Optional[BlockchainConnector] = None,
        wallet_address: Optional[str] = None,
        private_key: Optional[str] = None,
        client_id: str = None,
        ga_generations: int = 3,
        ga_population_size: int = 8
    ):
        """
        Initialize GA-Stacking client with scikit-learn models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            ensemble_size: Number of models in the ensemble
            ipfs_connector: IPFS connector
            blockchain_connector: Blockchain connector
            wallet_address: Client's wallet address
            private_key: Client's private key
            client_id: Client identifier
            ga_generations: Number of GA generations to run
            ga_population_size: Size of GA population
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.ensemble_size = ensemble_size
        self.ipfs = ipfs_connector or IPFSConnector()
        self.blockchain = blockchain_connector
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.client_id = client_id or f"client-{os.getpid()}"
        self.ga_generations = ga_generations
        self.ga_population_size = ga_population_size
        
        # Initialize GA-Stacking pipeline
        self.ga_pipeline = GAStackingPipeline(
            base_models=BASE_MODELS,
            meta_models=META_MODELS,
            generations=ga_generations,
            pop_size=ga_population_size,
            verbose=True
        )
        
        # Metrics storage
        self.metrics_history = []
        
        # Ensemble state
        self.ensemble_state = None
        self.ga_results = None
        
        logger.info(f"Initialized {self.client_id} with GA-Stacking pipeline")
        logger.info(f"IPFS node: {self.ipfs.ipfs_api_url}")
        
        # Verify blockchain authentication if available
        if self.blockchain and self.wallet_address:
            try:
                is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
                if is_authorized:
                    logger.info(f"Client {self.wallet_address} is authorized on the blockchain ✅")
                else:
                    logger.warning(f"Client {self.wallet_address} is NOT authorized on the blockchain ❌")
                    logger.warning("The server may reject this client's contributions")
            except Exception as e:
                logger.error(f"Failed to verify client authorization: {e}")
    
    # Rest of the GAStackingClient class remains unchanged...
    def get_parameters(self, config):
        """Get model parameters as a list of NumPy arrays."""
        if not hasattr(self, 'ensemble_state') or self.ensemble_state is None:
            # If ensemble not trained yet, return empty parameters
            return [np.array([])]
        
        # Serialize ensemble state to JSON and convert to ndarray
        ensemble_bytes = json.dumps(self.ensemble_state).encode('utf-8')
        return [np.frombuffer(ensemble_bytes, dtype=np.uint8)]
    
    def set_parameters_from_ensemble_state(self, ensemble_state):
        """Set ensemble state from a dictionary."""
        self.ensemble_state = ensemble_state
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set parameters from a list of NumPy arrays.
        
        Args:
            parameters: List of parameter arrays
        """
        if len(parameters) == 1 and parameters[0].dtype == np.uint8:
            # Try to interpret as serialized ensemble state
            try:
                ensemble_bytes = parameters[0].tobytes()
                ensemble_state = json.loads(ensemble_bytes.decode('utf-8'))
                self.set_parameters_from_ensemble_state(ensemble_state)
                logger.info("Successfully loaded ensemble state from parameters")
                return
            except Exception as e:
                logger.error(f"Failed to deserialize ensemble state: {e}")
        
        logger.warning("Failed to set parameters - invalid format")
    
    def set_parameters_from_ipfs(self, ipfs_hash: str) -> None:
        """Set model parameters from IPFS."""
        try:
            # Get model data from IPFS
            model_data = self.ipfs.get_json(ipfs_hash)
            
            if model_data and "ensemble_state" in model_data:
                # Load ensemble state
                ensemble_state = model_data["ensemble_state"]
                self.ensemble_state = ensemble_state
                logger.info(f"Ensemble model loaded from IPFS: {ipfs_hash}")
            else:
                logger.error(f"Invalid model data from IPFS: {ipfs_hash}")
                
        except Exception as e:
            logger.error(f"Failed to load model from IPFS: {e}")
            logger.error(f"Error details: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def save_ensemble_to_ipfs(self, round_num, metrics=None):
        """Save ensemble model to IPFS."""
        if not hasattr(self, 'ensemble_state') or self.ensemble_state is None:
            logger.warning("No ensemble state to save")
            return None
        
        try:
            # Safely create metadata with null checks
            model_metadata = {
                "ensemble_state": self.ensemble_state,
                "info": {
                    "round": round_num,
                    "client_id": self.client_id,
                    "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "ensemble_size": len(self.ensemble_state.get("model_names", [])),
                    "weights": self.ensemble_state.get("weights", [])
                }
            }
            
            # Add metrics if provided
            if metrics:
                model_metadata["info"]["metrics"] = metrics
            
            # Store in IPFS
            ipfs_hash = self.ipfs.add_json(model_metadata)
            logger.info(f"Stored ensemble model in IPFS: {ipfs_hash}")
            
            return ipfs_hash
        except Exception as e:
            logger.error(f"Failed to save ensemble to IPFS: {str(e)}")
            return None
    
    def fit(self, parameters: Parameters, config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Modified fit method to implement the desired workflow:
        - Round 1: Train base models + GA-Stacking with initial weights
        - Round 2+: Fine-tune with aggregated weights
        """
        # Check if client is authorized (if blockchain is available)
        if self.blockchain and self.wallet_address:
            try:
                is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
                if not is_authorized:
                    logger.warning(f"Client {self.wallet_address} is not authorized to participate")
                    # Return empty parameters with auth failure flag
                    return parameters_to_ndarrays(parameters), 0, {"error": "client_not_authorized"}
            except Exception as e:
                logger.error(f"Failed to check client authorization: {e}")
                # Continue anyway, server will check again

        # Get config parameters
        server_round = config.get("server_round", 0)
        is_first_round = server_round == 1
        global_model_ipfs_hash = config.get("global_model_ipfs_hash", None)
        
        # Create client-specific model directory
        model_dir = f"models/{self.client_id}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Use data splits (already prepared in your client)
        X_train, y_train = self.X_train, self.y_train
        X_val, y_val = self.X_val, self.y_val
        
        # Initialize ensemble weights to None (will be set based on round)
        initial_weights = None
        
        # Get ensemble weights from config or IPFS for both rounds
        if is_first_round:
            # For Round 1: Check if server provided initial weights in config
            if "initial_ensemble_weights" in config:
                initial_weights = config.get("initial_ensemble_weights")
                logger.info(f"Using server-provided initial weights: {initial_weights}")
        else:
            # For Rounds 2+: Get aggregated weights from IPFS
            if global_model_ipfs_hash:
                try:
                    # Retrieve aggregated ensemble weights from IPFS
                    global_model_data = self.ipfs.get_json(global_model_ipfs_hash)
                    
                    # Extract ensemble weights
                    initial_weights = global_model_data.get("ensemble_weights")
                    logger.info(f"Retrieved aggregated weights from IPFS: {initial_weights}")
                except Exception as e:
                    logger.error(f"Failed to retrieve ensemble weights from IPFS: {e}")
        
        # ROUND 1: Train base models + GA-Stacking (with optional initial weights)
        if is_first_round:
            logger.info("Round 1: Training base models with GA-Stacking")
            try:
                # Initialize the GA pipeline with config parameters
                self.ga_pipeline = GAStackingPipeline(
                    pop_size=config.get("pop_size", 50),
                    generations=config.get("generations", 20),
                    cv_folds=config.get("cv_folds", 5),
                    verbose=True
                )
                
                # Train the GA-Stacking pipeline, passing initial weights if available
                meta_X_train = generate_meta_features(
                    X_train, y_train, self.ga_pipeline.base_models, n_splits=self.ga_pipeline.cv_folds, n_jobs=-1
                )
                
                # Train base models
                with parallel_backend("loky"):
                    self.ga_pipeline.trained_base_models = train_base_models(
                        X_train, y_train, self.ga_pipeline.base_models, n_jobs=-1
                    )
                
                # Generate validation meta-features
                meta_X_val = np.column_stack([
                    model.predict_proba(X_val)[:, 1]
                    for model in self.ga_pipeline.trained_base_models.values()
                ])
                
                # Run GA with initial weights if provided
                self.ga_pipeline.best_weights, self.ga_pipeline.convergence_history = GA_weighted(
                    meta_X_train, y_train, meta_X_val, y_val,
                    pop_size=self.ga_pipeline.pop_size,
                    generations=self.ga_pipeline.generations,
                    crossover_prob=self.ga_pipeline.crossover_prob,
                    mutation_prob=self.ga_pipeline.mutation_prob,
                    metric=self.ga_pipeline.metric,
                    verbose=self.ga_pipeline.verbose,
                    init_weights=initial_weights  # Pass the initial weights here
                )
                
                # Get results (reusing your existing code)
                self.ga_results = {
                    "val_metrics": evaluate_metrics(y_val, ensemble_predict(meta_X_val, self.ga_pipeline.best_weights)),
                    "best_weights": self.ga_pipeline.best_weights.tolist(),
                    "model_names": self.ga_pipeline.model_names,
                    "diversity_score": self.ga_pipeline._calculate_diversity(X_val, y_val),
                    "generalization_score": self.ga_pipeline._calculate_generalization(X_train, y_train, X_val, y_val),
                    "convergence_history": self.ga_pipeline.convergence_history
                }
                
                # Save ensemble state
                self.ensemble_state = self.ga_pipeline.get_ensemble_state()
                
                # Save base models for future rounds
                self.ga_pipeline.save_base_models(model_dir)
                
                logger.info(f"Completed GA-Stacking with {len(self.ensemble_state.get('model_names', []))} base models")
            except Exception as e:
                logger.error(f"Error in GA-Stacking: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # ROUNDS 2+: Fine-tune with aggregated weights
        else:
            logger.info(f"Round {server_round}: Fine-tuning with aggregated weights")
            try:
                # Initialize the GA pipeline if not already done
                if not hasattr(self, 'ga_pipeline') or self.ga_pipeline is None:
                    self.ga_pipeline = GAStackingPipeline(verbose=True)
                
                # Load saved base models
                loaded_models = self.ga_pipeline.load_base_models(model_dir)
                
                # Perform fine-tuning with the aggregated weights
                self.ga_results = self.ga_pipeline.fine_tune(
                    X_train, y_train, X_val, y_val,
                    base_models=loaded_models,
                    weights=initial_weights  # Use the weights from IPFS
                )
                
                # Get updated ensemble state
                self.ensemble_state = self.ga_pipeline.get_ensemble_state()
                
                logger.info(f"Completed fine-tuning with weights")
            except Exception as e:
                logger.error(f"Error in fine-tuning: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Evaluate on test data (using your existing code)
        if hasattr(self, 'ga_pipeline') and self.ga_pipeline is not None and hasattr(self.ga_pipeline, 'predict'):
            # [Your existing evaluation code here]
            y_pred = self.ga_pipeline.predict(self.X_test)
            test_metrics = evaluate_metrics(self.y_test, y_pred)
            
            accuracy = test_metrics['accuracy'] * 100.0
            loss = 1.0 - test_metrics['precision']
            
            # Calculate base model average score
            if hasattr(self, 'ga_results') and self.ga_results:
                avg_base_model_score = np.mean([
                    test_metrics['f1'], 
                    self.ga_results.get('diversity_score', 0.0),
                    self.ga_results.get('generalization_score', 0.0)
                ])
            else:
                avg_base_model_score = 0.0
            
            # Prepare GA-Stacking metrics
            ga_stacking_metrics = {
                "data_size": len(self.X_train),
                "training_rounds": server_round,
                "evaluation_score": float(test_metrics['f1']),
                "ensemble_accuracy": float(test_metrics['accuracy']),
                "diversity_score": self.ga_results.get('diversity_score', 0.0) if hasattr(self, 'ga_results') else 0.0,
                "generalization_score": self.ga_results.get('generalization_score', 0.0) if hasattr(self, 'ga_results') else 0.0,
                "convergence_rate": self.ga_results.get('convergence_rate', 0.0) if hasattr(self, 'ga_results') else 0.0,
                "avg_base_model_score": avg_base_model_score,
                "final_score": (
                    float(test_metrics['f1']) * 0.4 +
                    self.ga_results.get('diversity_score', 0.0) * 0.2 +
                    self.ga_results.get('generalization_score', 0.0) * 0.2 +
                    self.ga_results.get('convergence_rate', 0.0) * 0.1 +
                    avg_base_model_score * 0.1
                ) if hasattr(self, 'ga_results') else float(test_metrics['f1'])
            }
            
            metrics = {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "precision": float(test_metrics['precision']),
                "recall": float(test_metrics['recall']),
                "f1_score": float(test_metrics['f1']),
            }
        else:
            # Fallback metrics if pipeline failed
            accuracy, loss = 0.0, float('inf')
            metrics = {
                "loss": float(loss),
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
            ga_stacking_metrics = {
                "data_size": len(self.X_train),
                "training_rounds": server_round,
                "evaluation_score": 0.0,
                "ensemble_accuracy": 0.0,
                "diversity_score": 0.0,
                "generalization_score": 0.0,
                "convergence_rate": 0.0,
                "avg_base_model_score": 0.0,
                "final_score": 0.0
            }
        
        # Store ensemble weights and metrics in IPFS
        try:
            client_data = {
                "ensemble_weights": self.ensemble_state.get("weights", []),
                "base_model_names": self.ensemble_state.get("model_names", []),
                "data_size": len(self.X_train),
                "metrics": {
                    "accuracy": float(accuracy),
                    "f1_score": float(metrics.get('f1_score', 0.0)),
                    "final_score": ga_stacking_metrics.get("final_score", 0.0)
                },
                "client_id": self.client_id,
                "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "round": server_round
            }
            
            client_ipfs_hash = self.ipfs.add_json(client_data)
            logger.info(f"Stored client data in IPFS: {client_ipfs_hash}")
        except Exception as e:
            logger.error(f"Failed to save client data to IPFS: {str(e)}")
            client_ipfs_hash = None
        
        # Update metrics history
        self.metrics_history.append({
            "round": server_round,
            "fit_loss": float(loss),
            "accuracy": float(accuracy),
            "ipfs_hash": client_ipfs_hash,
            "wallet_address": self.wallet_address if self.wallet_address else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ga_stacking_metrics": ga_stacking_metrics
        })
        
        # Include necessary information in metrics
        metrics["client_ipfs_hash"] = client_ipfs_hash
        metrics["wallet_address"] = self.wallet_address if self.wallet_address else "unknown"
        metrics["ensemble_size"] = len(self.ensemble_state.get("model_names", [])) if hasattr(self, 'ensemble_state') else 0
        metrics["base_model_names"] = self.ensemble_state.get("model_names", []) if hasattr(self, 'ensemble_state') else []
        metrics["ensemble_weights"] = self.ensemble_state.get("weights", []) if hasattr(self, 'ensemble_state') else []
        
        # Add GA-Stacking metrics
        for key, value in ga_stacking_metrics.items():
            metrics[key] = value
        
        # Return empty parameters with metrics
        empty_parameters = [np.array([])]
        
        # Ensure serializable metrics
        for key in list(metrics.keys()):
            if not isinstance(metrics[key], (int, float, str, bool, list)):
                metrics[key] = str(metrics[key])
        
        return empty_parameters, len(self.X_train), metrics
    
    def evaluate(self, parameters: Parameters, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on the local test dataset using GA-Stacking.
        Modified to handle the case when the model is not yet trained.
        """
        # Check if client is authorized (if blockchain is available)
        if self.blockchain and self.wallet_address:
            try:
                is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
                if not is_authorized:
                    logger.warning(f"Client {self.wallet_address} is not authorized to participate in evaluation")
                    # Return empty evaluation with auth failure flag
                    return 0.0, 0, {"error": "client_not_authorized"}
            except Exception as e:
                logger.error(f"Failed to check client authorization: {e}")
        
        # Ensure parameters are in the correct format
        if isinstance(parameters, list):
            # Parameters are already a list of NumPy arrays
            params = parameters
        else:
            # Convert Parameters object to a list of NumPy arrays
            params = parameters_to_ndarrays(parameters)
        
        # Get global model from IPFS if hash is provided
        ipfs_hash = config.get("ipfs_hash", None)
        round_num = config.get("server_round", 0)
        
        # Set parameters if provided
        if len(params) > 0 and len(params[0]) > 0:
            self.set_parameters(params)
        
        # Check if the pipeline is trained and has the predict method
        pipeline_trained = (hasattr(self, 'ga_pipeline') and 
                        self.ga_pipeline is not None and 
                        hasattr(self.ga_pipeline, 'predict') and
                        hasattr(self, 'ensemble_state') and 
                        self.ensemble_state is not None)
        
        if not pipeline_trained:
            logger.warning("GA pipeline not trained yet, returning default evaluation metrics")
            # Return default metrics
            return float('inf'), len(self.X_test), {
                "loss": float('inf'),
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc_roc": 0.5,
                "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                "model_status": "not_trained"
            }
        
        # If the pipeline is trained, proceed with evaluation
        try:
            y_pred = self.ga_pipeline.predict(self.X_test)
            test_metrics = evaluate_metrics(self.y_test, y_pred)
            
            accuracy = test_metrics['accuracy'] * 100.0
            loss = 1.0 - test_metrics['auc']  # Use 1-AUC as a loss
            
            # Calculate confusion matrix components
            threshold = 0.3  # Adjust based on your use case
            y_pred_labels = (y_pred > threshold).astype(int)
            
            # Use sklearn's confusion_matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(self.y_test, y_pred_labels)
            
            # Handle both binary and multi-class cases
            if cm.shape == (2, 2):  # Binary classification
                tn, fp, fn, tp = cm.ravel()
            else:  # Multi-class - calculate based on one-vs-rest
                # For fraud detection, usually class 1 is fraud
                fraud_idx = 1
                tp = cm[fraud_idx, fraud_idx]
                fp = cm[:, fraud_idx].sum() - tp
                fn = cm[fraud_idx, :].sum() - tp
                tn = cm.sum() - (tp + fp + fn)
            
            metrics = {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "precision": float(test_metrics['precision']),
                "recall": float(test_metrics['recall']),
                "f1_score": float(test_metrics['f1']),
                "auc_roc": float(test_metrics['auc']),
                "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                "ensemble_size": len(self.ensemble_state.get("model_names", [])) if hasattr(self, 'ensemble_state') else 0,
                
                # Add confusion matrix components
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                
                "model_status": "trained"
            }
            
            # Add GA-Stacking metrics if available
            if hasattr(self, 'ga_results'):
                metrics.update({
                    "ensemble_accuracy": float(self.ga_results["val_metrics"]["accuracy"]),
                    "diversity_score": float(self.ga_results["diversity_score"]),
                    "generalization_score": float(self.ga_results["generalization_score"]),
                    "convergence_rate": float(self.ga_results["convergence_rate"])
                })
            
            # Add to metrics history
            self.metrics_history.append({
                "round": round_num,
                "eval_loss": float(loss),
                "accuracy": float(accuracy),
                "precision": float(test_metrics['precision']),
                "recall": float(test_metrics['recall']),
                "f1_score": float(test_metrics['f1']),
                "auc_roc": float(test_metrics['auc']),
                "eval_samples": len(self.X_test),
                "ipfs_hash": ipfs_hash,
                "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return float(loss), len(self.X_test), metrics
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            # Return default metrics in case of error
            return float('inf'), len(self.X_test), {
                "loss": float('inf'),
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc_roc": 0.5,
                "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                "error": str(e),
                "model_status": "error"
        }
    
    def save_metrics_history(self, filepath: str = "client_metrics.json"):
        """Save metrics history to a file."""
        with open(filepath, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Saved metrics history to {filepath}")


def load_csv_dataset(csv_file_path: str, target_col: str = "Class") -> pd.DataFrame:
    """
    Load a dataset from a CSV file.
    
    Args:
        csv_file_path: Path to the CSV file
        target_col: Name of the target column
        
    Returns:
        DataFrame with the loaded data
    """
    try:
        # Check if file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_file_path}")
        
        # Load the CSV file
        data = pd.read_csv(csv_file_path)
        
        # Ensure the target column exists
        if target_col not in data.columns:
            if 'Class' in data.columns:
                target_col = 'Class'
                logger.warning(f"Target column '{target_col}' not found, using 'Class' instead")
            else:
                # Try to identify the target column based on common names
                potential_targets = ['target', 'label', 'y', 'class', 'fraud', 'is_fraud']
                for col in potential_targets:
                    if col in data.columns:
                        target_col = col
                        logger.warning(f"Target column not found, using '{target_col}' instead")
                        break
                else:
                    # If no known target column is found, use the last column
                    target_col = data.columns[-1]
                    logger.warning(f"Target column not found, using last column '{target_col}' as target")
        
        # Check if 'Time' column exists and add it if not
        if 'Time' not in data.columns:
            data['Time'] = range(len(data))
            logger.info("Added 'Time' column to dataset")
            
        logger.info(f"Successfully loaded dataset with {len(data)} samples and {len(data.columns)} features")
        logger.info(f"Target column: {target_col}, Class distribution: {data[target_col].value_counts().to_dict()}")
        
        return data
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def start_client(
    server_address: str = "192.168.80.180:8088",
    ipfs_url: str = "http://127.0.0.1:5001",
    ganache_url: str = "http://127.0.0.1:7545",
    contract_address: Optional[str] = None,
    wallet_address: Optional[str] = None,
    private_key: Optional[str] = None,
    client_id: Optional[str] = None,
    ensemble_size: int = 8,
    ga_generations: int = 3,
    ga_population_size: int = 8,
    csv_file: Optional[str] = None,
    target_column: str = "Class",
    test_size: float = 0.2
) -> None:
    """
    Start a federated learning client with GA-Stacking ensemble optimization.
    
    Args:
        server_address: Server address (host:port)
        ipfs_url: IPFS API URL
        ganache_url: Ganache blockchain URL
        contract_address: Address of deployed EnhancedModelRegistry contract
        wallet_address: Client's Ethereum wallet address
        private_key: Client's private key (for signing transactions)
        client_id: Client identifier
        ensemble_size: Number of models in the ensemble
        ga_generations: Number of GA generations to run
        ga_population_size: Size of GA population
        csv_file: Path to the CSV data file
        target_column: Name of the target column in the CSV file
        test_size: Proportion of data to use for testing
    """
    # Extract client ID from CSV file path if not explicitly provided
    if client_id is None and csv_file is not None:
        # Try to extract client ID from the filename
        filename = os.path.basename(csv_file)
        if filename.startswith("client_"):
            # Extract client_X from filename like "client_1.csv"
            client_id = filename.split(".")[0]
            logger.info(f"Extracted client ID from CSV file: {client_id}")
        else:
            # Use a default client ID with timestamp to avoid collisions
            timestamp = int(time.time())
            client_id = f"client_{timestamp}"
            logger.info(f"Using generated client ID: {client_id}")
    elif client_id is None:
        # Default client ID if no CSV file or explicit ID provided
        client_id = f"client_{os.getpid()}"
        logger.info(f"Using process-based client ID: {client_id}")
    
    # Create metrics directory
    os.makedirs(f"metrics/{client_id}", exist_ok=True)
    
    # Initialize IPFS connector
    ipfs_connector = IPFSConnector(ipfs_api_url=ipfs_url)
    logger.info(f"Initialized IPFS connector: {ipfs_url}")
    
    # Initialize blockchain connector if contract address is provided
    blockchain_connector = None
    if contract_address:
        try:
            blockchain_connector = BlockchainConnector(
                ganache_url=ganache_url,
                contract_address=contract_address,
                private_key=private_key
            )
            logger.info(f"Initialized blockchain connector: {ganache_url}")
            logger.info(f"Using contract at: {contract_address}")
        except Exception as e:
            logger.error(f"Failed to initialize blockchain connector: {e}")
            logger.warning("Continuing without blockchain features")
    
    # Determine dataset file to load
    if csv_file is None:
        # If file isn't explicitly provided, derive from client ID
        derived_csv_file = f"{client_id}.csv"
        
        # Check if derived file exists
        if not os.path.exists(derived_csv_file):
            logger.warning(f"Derived dataset file not found: {derived_csv_file}")
            
            # Try to find in data directory
            data_dir_csv = f"data/{client_id}.csv"
            
            if os.path.exists(data_dir_csv):
                logger.info(f"Found dataset file in data directory")
                csv_file = data_dir_csv
            else:
                # Fall back to default client_0.csv file
                logger.warning(f"Using default dataset file")
                csv_file = "client_0.csv"
                
                # Check if default file exists in data directory
                if not os.path.exists(csv_file):
                    data_dir_default_csv = "data/client_0.csv"
                    
                    if os.path.exists(data_dir_default_csv):
                        csv_file = data_dir_default_csv
                    else:
                        raise FileNotFoundError(f"No dataset file found for {client_id}")
        else:
            csv_file = derived_csv_file
    
    logger.info(f"Using dataset file: {csv_file}")
    
    # Load and process dataset
    try:
        # Load CSV data
        data = load_csv_dataset(csv_file, target_column)
        
        # Use split_and_scale function to get train/val/test splits
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(
            data=data,
            target_col=target_column,
            test_size=test_size,
            random_state=42
        )
        
        logger.info(f"Data preprocessing complete:")
        logger.info(f"- Training set: {X_train.shape[0]} samples")
        logger.info(f"- Validation set: {X_val.shape[0]} samples")
        logger.info(f"- Test set: {X_test.shape[0]} samples")
        
        # For client interface, combine train and validation for X_train, y_train
        # The client's fit() method will split them again for GA-Stacking
        #X_train_combined = np.vstack([X_train, X_val])
        #y_train_combined = np.concatenate([y_train, y_val])
        
        #logger.info(f"Combined training set: {X_train_combined.shape[0]} samples")
        
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        raise
    
    # Create client
    client = GAStackingClient(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        ensemble_size=ensemble_size,
        ipfs_connector=ipfs_connector,
        blockchain_connector=blockchain_connector,
        wallet_address=wallet_address,
        private_key=private_key,
        client_id=client_id,
        ga_generations=ga_generations,
        ga_population_size=ga_population_size
    )
    
    # Start client
    fl.client.start_client(server_address=server_address, client=client)
    
    # Save metrics after client finishes
    client.save_metrics_history(filepath=f"metrics/{client_id}/metrics_history.json")
    
    logger.info(f"Client {client_id} completed federated learning with GA-Stacking")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start FL client with GA-Stacking ensemble optimization")
    parser.add_argument("--server-address", type=str, default="192.168.80.180:8088", help="Server address (host:port)")
    parser.add_argument("--ipfs-url", type=str, default="http://127.0.0.1:5001/api/v0", help="IPFS API URL")
    parser.add_argument("--ganache-url", type=str, default="http://192.168.80.1:7545", help="Ganache blockchain URL")
    parser.add_argument("--contract-address", type=str, help="Address of deployed EnhancedModelRegistry contract")
    parser.add_argument("--wallet-address", type=str, help="Client's Ethereum wallet address")
    parser.add_argument("--private-key", type=str, help="Client's private key (for signing transactions)")
    parser.add_argument("--client-id", type=str, help="Client identifier")
    parser.add_argument("--ensemble-size", type=int, default=8, help="Number of models in the ensemble")
    parser.add_argument("--ga-generations", type=int, default=3, help="Number of GA generations to run")
    parser.add_argument("--ga-population-size", type=int, default=10, help="Size of GA population")
    parser.add_argument("--csv-file", type=str, help="Path to CSV data file")
    parser.add_argument("--target-column", type=str, default="Class", help="Name of target column in CSV file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data to use for testing")
    
    # For backward compatibility
    parser.add_argument("--train-file", type=str, help="[DEPRECATED] Path to training data file")
    parser.add_argument("--test-file", type=str, help="[DEPRECATED] Path to test data file")
    args = parser.parse_args()
    
    # Print warning if using deprecated arguments
    if args.train_file or args.test_file:
        print("Warning: --train-file and --test-file are deprecated. Please use --csv-file instead.")
        print("Check our documentation for information on using CSV files with the updated client.")
    
    # Check if contract address is stored in file
    if args.contract_address is None:
        try:
            with open("contract_address.txt", "r") as f:
                args.contract_address = f.read().strip()
                print(f"Loaded contract address from file: {args.contract_address}")
        except FileNotFoundError:
            print("No contract address provided or found in file")
    
    # Handle backward compatibility for train/test files
    csv_file = args.csv_file
    if csv_file is None and args.train_file:
        print(f"Using train file in backward compatibility mode: {args.train_file}")
        # Here you would add logic to convert train/test files to a single CSV
        # For now, just print a warning
        print("Please convert your data to CSV format for best compatibility.")
    
    start_client(
        server_address=args.server_address,
        ipfs_url=args.ipfs_url,
        ganache_url=args.ganache_url,
        contract_address=args.contract_address,
        wallet_address=args.wallet_address,
        private_key=args.private_key,
        client_id=args.client_id,
        ensemble_size=args.ensemble_size,
        ga_generations=args.ga_generations,
        ga_population_size=args.ga_population_size,
        csv_file=args.csv_file,
        target_column=args.target_column,
        test_size=args.test_size
    )
