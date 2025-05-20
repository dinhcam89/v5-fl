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
    confusion_matrix
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
logger = logging.getLogger("GAStacking-FL-Client")


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
        self.ipfs = None
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

    def fit(self, parameters: Parameters, config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train the model on the local dataset using GA-Stacking.
        Modified to handle different rounds: full training in round 1, fine-tuning in later rounds.
        """
        # Check if client is authorized (if blockchain is available)
        # if self.blockchain and self.wallet_address:
        #     try:
        #         is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
        #         if not is_authorized:
        #             logger.warning(f"Client {self.wallet_address} is not authorized to participate")
        #             # Return empty parameters with auth failure flag
        #             return parameters_to_ndarrays(parameters), 0, {"error": "client_not_authorized"}
        #     except Exception as e:
        #         logger.error(f"Failed to check client authorization: {e}")
        #         # Continue anyway, server will check again

        # Get round number
        round_num = config.get("server_round", 0)
        
        # Get training config
        do_ga_stacking = bool(config.get("ga_stacking", True))
        validation_split = float(config.get("validation_split", 0.3))
        
        # Create validation split
        X_train, y_train = self.X_train, self.y_train
        X_val, y_val = self.X_val, self.y_val
        
        # Define base model storage path
        base_models_path = os.path.join(f"./client_storage/{self.client_id}/base_models")
        os.makedirs(base_models_path, exist_ok=True)
        
        # Different logic for Round 1 vs. later rounds
        if round_num == 1:
            # ROUND 1: Full training of base models + GA-Stacking
            logger.info(f"Round {round_num}: Performing full training of base models + GA-Stacking")
            
            if do_ga_stacking:
                try:
                    # Train the GA-Stacking pipeline from scratch
                    self.ga_results = self.ga_pipeline.train(X_train, y_train, X_val, y_val)
                    
                    # Store the ensemble state
                    self.ensemble_state = self.ga_pipeline.get_ensemble_state()
                    
                    # Save base models for future rounds
                    try:
                        self.ga_pipeline.save_base_models(base_models_path)
                        logger.info(f"Base models saved for future rounds at {base_models_path}")
                    except Exception as save_err:
                        logger.error(f"Failed to save base models: {save_err}")
                    
                    # Extract GA-Stacking metrics - SEND RAW METRICS TO SERVER
                    ga_stacking_metrics = {
                        "ensemble_accuracy": self.ga_results["val_metrics"]["accuracy"],
                        "diversity_score": self.ga_results["diversity_score"],
                        "generalization_score": self.ga_results["generalization_score"],
                        "convergence_rate": self.ga_results["convergence_rate"],
                        "avg_base_model_score": self.ga_results.get("avg_base_model_score", 0.0),
                        "training_mode": "full_training"  # Indicate this was full training
                    }
                    
                    logger.info(f"GA-Stacking complete - Accuracy: {ga_stacking_metrics['ensemble_accuracy']:.4f}, "
                            f"Diversity: {ga_stacking_metrics['diversity_score']:.4f}, "
                            f"Generalization: {ga_stacking_metrics['generalization_score']:.4f}, "
                            f"Convergence: {ga_stacking_metrics['convergence_rate']:.4f}, "
                            f"Avg Base Model Score: {ga_stacking_metrics['avg_base_model_score']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error in GA-Stacking full training: {str(e)}")
                    # Fallback to simple ensemble
                    ga_stacking_metrics = {
                        "ensemble_accuracy": 0.0,
                        "diversity_score": 0.0,
                        "generalization_score": 0.0,
                        "convergence_rate": 0.5,
                        "avg_base_model_score": 0.0,
                        "training_mode": "full_training_failed"
                    }
            else:
                logger.info("Skipping GA-Stacking as requested in config")
                ga_stacking_metrics = {
                    "ensemble_accuracy": 0.0,
                    "diversity_score": 0.0,
                    "generalization_score": 0.0,
                    "convergence_rate": 0.5,
                    "avg_base_model_score": 0.0,
                    "training_mode": "skipped"
                }
        else:
            # ROUNDS 2-N: Fine-tuning with existing base models
            logger.info(f"Round {round_num}: Fine-tuning existing base models")
            try:
                 # Load base models from storage
                loaded_models = self.ga_pipeline.load_base_models(base_models_path)
                logger.info(f"Loaded {len(loaded_models)} base models for fine-tuning")
                
                # Try to extract server ensemble state from parameters
                server_ensemble_state = None
                try:
                    params = parameters_to_ndarrays(parameters)
                    if len(params) == 1 and params[0].dtype == np.uint8:
                        ensemble_bytes = params[0].tobytes()
                        server_ensemble_state = json.loads(ensemble_bytes.decode('utf-8'))
                        logger.info(f"Successfully loaded server ensemble state with {len(server_ensemble_state.get('model_names', []))} models")
                except Exception as e:
                    logger.warning(f"Could not extract ensemble state from parameters: {e}")
                
                # Fine-tune the models with server ensemble state if available
                self.ga_results = self.ga_pipeline.fine_tune(
                    X_train, y_train, X_val, y_val, 
                    base_models=loaded_models,
                    server_ensemble_state=server_ensemble_state
                )
                self.ensemble_state = self.ga_pipeline.get_ensemble_state()
                ga_stacking_metrics = {
                    "ensemble_accuracy": self.ga_results["val_metrics"]["accuracy"],
                    "diversity_score": self.ga_results["diversity_score"],
                    "generalization_score": self.ga_results["generalization_score"],
                    "convergence_rate": self.ga_results["convergence_rate"],
                    "avg_base_model_score": self.ga_results.get("avg_base_model_score", 0.0),
                    "training_mode": "fine_tuning"
                }
                logger.info(f"Fine-tuning complete - Accuracy: {ga_stacking_metrics['ensemble_accuracy']:.4f}, "
                            f"Diversity: {ga_stacking_metrics['diversity_score']:.4f}, "
                            f"Generalization: {ga_stacking_metrics['generalization_score']:.4f}")
            except Exception as e:
                logger.error(f"Error in fine-tuning: {str(e)}")
                # Fallback - try full training if fine-tuning fails
                try:
                    logger.warning("Fine-tuning failed, falling back to full training")
                    self.ga_results = self.ga_pipeline.train(X_train, y_train, X_val, y_val)
                    self.ensemble_state = self.ga_pipeline.get_ensemble_state()
                    
                    # Save models for next time
                    self.ga_pipeline.save_base_models(base_models_path)
                    
                    ga_stacking_metrics = {
                        "ensemble_accuracy": self.ga_results["val_metrics"]["accuracy"],
                        "diversity_score": self.ga_results["diversity_score"],
                        "generalization_score": self.ga_results["generalization_score"],
                        "convergence_rate": self.ga_results["convergence_rate"],
                        "avg_base_model_score": self.ga_results.get("avg_base_model_score", 0.0),
                        "training_mode": "fallback_full_training"  # Indicate fallback to full training
                    }
                except Exception as fallback_err:
                    logger.error(f"Fallback to full training also failed: {fallback_err}")
                    ga_stacking_metrics = {
                        "ensemble_accuracy": 0.0,
                        "diversity_score": 0.0,
                        "generalization_score": 0.0,
                        "convergence_rate": 0.5,
                        "avg_base_model_score": 0.0,
                        "training_mode": "training_failed"
                    }

        # Evaluate on test data if we have a trained pipeline
        if hasattr(self, 'ga_pipeline') and self.ga_pipeline is not None and hasattr(self.ga_pipeline, 'predict'):
            y_pred = self.ga_pipeline.predict(self.X_test)
            test_metrics = evaluate_metrics(self.y_test, y_pred)
            
            accuracy = test_metrics['accuracy'] * 100.0
            loss = 1.0 - test_metrics['precision']  # Use 1 - precision as loss
            
            metrics = {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "precision": float(test_metrics['precision']),
                "recall": float(test_metrics['recall']),
                "f1_score": float(test_metrics['f1'])
            }
        else:
            # Fallback if pipeline isn't trained
            accuracy, loss = 0.0, float('inf')
            metrics = {
                "loss": float(loss),
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }

        # # Save ensemble to IPFS
        # client_ipfs_hash = None
        # try:
        #     # Create metadata with ensemble state and round info
        #     model_metadata = {
        #         "ensemble_state": self.ensemble_state,
        #         "info": {
        #             "round": round_num,
        #             "client_id": self.client_id,
        #             "wallet_address": self.wallet_address if self.wallet_address else "unknown",
        #             "timestamp": datetime.now(timezone.utc).isoformat(),
        #             "ensemble_size": len(self.ensemble_state.get("model_names", [])),
        #             "weights": self.ensemble_state.get("weights", []),
        #             "metrics": metrics,
        #             "training_mode": ga_stacking_metrics.get("training_mode", "unknown")
        #         }
        #     }
            
        #     # Store in IPFS with retries
        #     for attempt in range(3):  # Try up to 3 times
        #         try:
        #             client_ipfs_hash = self.ipfs.add_json(model_metadata)
        #             if client_ipfs_hash:
        #                 logger.info(f"Stored ensemble model in IPFS: {client_ipfs_hash}")
        #                 break
        #             else:
        #                 logger.warning(f"Failed to get valid IPFS hash (attempt {attempt+1}/3)")
        #                 time.sleep(1)  # Wait before retry
        #         except Exception as retry_err:
        #             logger.warning(f"IPFS storage attempt {attempt+1}/3 failed: {retry_err}")
        #             time.sleep(2)  # Wait longer before retry
                
        #     if not client_ipfs_hash:
        #         raise ValueError("Failed to get valid IPFS hash after multiple attempts")
                
        # except Exception as e:
        #     logger.error(f"Failed to save ensemble to IPFS: {str(e)}")
        #     # If IPFS fails, we have no model to return - this is critical in IPFS-only mode
        #     # Return a flag indicating IPFS failure
        #     dummy_parameters = parameters_to_ndarrays(parameters)
        #     return dummy_parameters, 0, {"error": "ipfs_storage_failed", "message": str(e)}

        # Add to metrics history
        self.metrics_history.append({
            "round": round_num,
            "fit_loss": float(loss),
            "accuracy": float(accuracy),
            "wallet_address": self.wallet_address if self.wallet_address else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ga_stacking_metrics": ga_stacking_metrics,
            "training_mode": ga_stacking_metrics.get("training_mode", "unknown")
        })

        # Include additional info in metrics
        metrics["wallet_address"] = self.wallet_address if self.wallet_address else "unknown"
        metrics["ensemble_size"] = len(self.ensemble_state.get("model_names", [])) if (hasattr(self, 'ensemble_state') and self.ensemble_state is not None) else 0
        metrics["model_transfer_mode"] = "direct"  # Flag to inform server we're using IPFS-only mode
        metrics["training_mode"] = ga_stacking_metrics.get("training_mode", "unknown")
        
        # Add GA-Stacking raw metrics to the returned metrics WITHOUT score calculation
        for key, value in ga_stacking_metrics.items():
            metrics[key] = value
            
        # Check for None values in metrics
        for key, value in metrics.items():
            if value is None:
                metrics[key] = 0.0  # Replace None with default value
                logger.warning(f"Replaced None value for metric '{key}' with 0.0")

        # For all rounds, serialize ensemble state to parameters and return
        if hasattr(self, 'ensemble_state') and self.ensemble_state is not None:
            ensemble_bytes = json.dumps(self.ensemble_state).encode('utf-8')
            ensemble_array = np.frombuffer(ensemble_bytes, dtype=np.uint8)
            return [ensemble_array], len(self.X_train), metrics
        else:
            # Fallback if no ensemble state
            placeholder_parameters = [np.array([ord('E'), ord('M'), ord('P'), ord('T'), ord('Y')], dtype=np.uint8)]
            return placeholder_parameters, len(self.X_train), metrics
    
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
                "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                "model_status": "not_trained"
            }
        
        # If the pipeline is trained, proceed with evaluation
        try:
            y_pred = self.ga_pipeline.predict(self.X_test)
            test_metrics = evaluate_metrics(self.y_test, y_pred)
            
            accuracy = test_metrics['accuracy'] * 100.0
            loss = 1.0 - test_metrics['f1']
                        
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
    ipfs_connector = None
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
    max_reconnect_attempts = 5
    current_attempt = 0
    backoff_time = 3  # seconds
    
    while current_attempt < max_reconnect_attempts:
        try:
            logger.info(f"Connecting to server (attempt {current_attempt + 1}/{max_reconnect_attempts})...")
            
            # Start client with reconnect=True
            fl.client.start_client(
                server_address=server_address, 
                client=client.to_client()
            )
            
            # If we get here, the client has gracefully disconnected
            logger.info("Client completed all rounds successfully")
            break
            
        except Exception as e:
            current_attempt += 1
            logger.error(f"Connection error (attempt {current_attempt}/{max_reconnect_attempts}): {str(e)}")
            
            if current_attempt < max_reconnect_attempts:
                wait_time = backoff_time * (2 ** (current_attempt - 1))  # Exponential backoff
                logger.info(f"Reconnecting in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Max reconnection attempts reached. Exiting.")
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
