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
from ga_stacking.utils import evaluate_metrics

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
        Train the model on the local dataset using GA-Stacking.
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

        # Get IPFS hash and round number
        ipfs_hash = config.get("ipfs_hash", None)
        round_num = config.get("server_round", 0)

        # Get training config
        do_ga_stacking = bool(config.get("ga_stacking", True))
        validation_split = float(config.get("validation_split", 0.2))
        
        # Create validation split
        try:
            # Combine training and test data into a single DataFrame for splitting
            data = pd.DataFrame(self.X_train, columns=[f"feature_{i}" for i in range(self.X_train.shape[1])])
            data["Class"] = self.y_train  # Add the target column

            # Use split_and_scale to split and preprocess the data
            X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(
                data=data,
                target_col="Class",  # Adjust if your target column has a different name
                test_size=validation_split,
                random_state=42
            )
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise
        
        # Run GA-Stacking
        if do_ga_stacking:
            logger.info("Performing GA-Stacking optimization")
            try:
                # Train the GA-Stacking pipeline
                self.ga_results = self.ga_pipeline.train(X_train, y_train, X_val, y_val)
                
                # Store the ensemble state
                self.ensemble_state = self.ga_pipeline.get_ensemble_state()
                
                # Extract GA-Stacking metrics
                ga_stacking_metrics = {
                    "ensemble_accuracy": self.ga_results["val_metrics"]["accuracy"],
                    "diversity_score": self.ga_results["diversity_score"],
                    "generalization_score": self.ga_results["generalization_score"],
                    "convergence_rate": self.ga_results["convergence_rate"],
                    "avg_base_model_score": self.ga_results.get("avg_base_model_score", 0.0),
                }
                
                # Calculate final score
                weights = {
                    "ensemble_accuracy": 0.40,
                    "diversity_score": 0.20,
                    "generalization_score": 0.20,
                    "convergence_rate": 0.10,
                    "avg_base_model_score": 0.10
                }
                
                weighted_score = sum(ga_stacking_metrics[key] * weight for key, weight in weights.items())
                
                # Apply bonus for exceptional accuracy (>90%)
                if ga_stacking_metrics["ensemble_accuracy"] > 0.9:
                    bonus = (ga_stacking_metrics["ensemble_accuracy"] - 0.9) * 1.5
                    weighted_score += bonus
                    ga_stacking_metrics["accuracy_bonus"] = float(bonus)
                
                # Convert to integer score (0-10000)
                ga_stacking_metrics["final_score"] = int(min(1.0, weighted_score) * 10000)
                
                logger.info(f"GA-Stacking complete - Accuracy: {ga_stacking_metrics['ensemble_accuracy']:.4f}, "
                        f"Diversity: {ga_stacking_metrics['diversity_score']:.4f}, "
                        f"Final Score: {ga_stacking_metrics['final_score']}")
                
            except Exception as e:
                logger.error(f"Error in GA-Stacking: {str(e)}")
                # Fallback to simple ensemble
                ga_stacking_metrics = {
                    "ensemble_accuracy": 0.0,
                    "diversity_score": 0.0,
                    "generalization_score": 0.0,
                    "convergence_rate": 0.5,
                    "avg_base_model_score": 0.0,
                    "final_score": 0
                }
        else:
            logger.info("Skipping GA-Stacking as requested in config")
            ga_stacking_metrics = {
                "ensemble_accuracy": 0.0,
                "diversity_score": 0.0,
                "generalization_score": 0.0,
                "convergence_rate": 0.5,
                "avg_base_model_score": 0.0,
                "final_score": 0
            }

        # Evaluate on test data
        # If we have a trained pipeline, use it for evaluation
        if hasattr(self, 'ga_pipeline') and self.ga_pipeline is not None and hasattr(self.ga_pipeline, 'predict'):

            y_pred = self.ga_pipeline.predict(self.X_test)
            test_metrics = evaluate_metrics(self.y_test, y_pred)
            
            accuracy = test_metrics['accuracy'] * 100.0
            loss = 1.0 - test_metrics['auc']  # Use 1-AUC as a loss
            
            metrics = {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "precision": float(test_metrics['precision']),
                "recall": float(test_metrics['recall']),
                "f1_score": float(test_metrics['f1']),
                "auc_roc": float(test_metrics['auc']),
            }
        else:
            # Fallback if pipeline isn't trained
            accuracy, loss = 0.0, float('inf')
            metrics = {
                "loss": float(loss),
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc_roc": 0.5,
            }

        # Save ensemble to IPFS
        try:
            # Create metadata
            model_metadata = {
                "ensemble_state": self.ensemble_state,
                "info": {
                    "round": round_num,
                    "client_id": self.client_id,
                    "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "ensemble_size": len(self.ensemble_state.get("model_names", [])),
                    "weights": self.ensemble_state.get("weights", []),
                    "metrics": metrics
                }
            }
            
            # Store in IPFS
            client_ipfs_hash = self.ipfs.add_json(model_metadata)
            logger.info(f"Stored ensemble model in IPFS: {client_ipfs_hash}")
        except Exception as e:
            logger.error(f"Failed to save ensemble to IPFS: {str(e)}")
            client_ipfs_hash = None

        # Add to metrics history
        self.metrics_history.append({
            "round": round_num,
            "fit_loss": float(loss),
            "accuracy": float(accuracy),
            "ipfs_hash": client_ipfs_hash,
            "wallet_address": self.wallet_address if self.wallet_address else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ga_stacking_metrics": ga_stacking_metrics
        })

        # Include additional info in metrics
        metrics["ipfs_hash"] = ipfs_hash  # Return the server's hash for tracking
        metrics["client_ipfs_hash"] = client_ipfs_hash
        metrics["wallet_address"] = self.wallet_address if self.wallet_address else "unknown"
        metrics["ensemble_size"] = len(self.ensemble_state.get("model_names", [])) if (hasattr(self, 'ensemble_state') and self.ensemble_state is not None) else 0
        
        # Add GA-Stacking specific metrics to the returned metrics
        for key, value in ga_stacking_metrics.items():
            metrics[key] = value

        # Get the ensemble parameters
        try:
            # Serialize ensemble state to JSON and convert to ndarray
            ensemble_bytes = json.dumps(self.ensemble_state).encode('utf-8')
            parameters_updated = [np.array(list(ensemble_bytes), dtype=np.uint8)]

        except Exception as e:
            logger.error(f"Error getting parameters: {str(e)}")
            parameters_updated = parameters_to_ndarrays(parameters)  # Return original parameters

        # Get number of training samples
        num_samples = len(self.X_train)

        # Make sure metrics only contains serializable values
        for key in list(metrics.keys()):
            if not isinstance(metrics[key], (int, float, str, bool, list)):
                metrics[key] = str(metrics[key])

        # Return the raw parameters (not wrapped in Flower Parameters)
        return parameters_updated, num_samples, metrics
    
    def evaluate(self, parameters: Parameters, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on the local test dataset using GA-Stacking.
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
        
        # If we have a trained pipeline, use it for evaluation
        if hasattr(self, 'ga_pipeline') and self.ga_pipeline is not None and hasattr(self.ga_pipeline, 'predict'):
            y_pred = self.ga_pipeline.predict(self.X_test)
            test_metrics = evaluate_metrics(self.y_test, y_pred)
            
            accuracy = test_metrics['accuracy'] * 100.0
            loss = 1.0 - test_metrics['auc']  # Use 1-AUC as a loss
            
            # IMPORTANT ADDITION: Calculate confusion matrix components
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
                "false_negatives": int(fn)
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
        else:
            # Fallback if pipeline isn't trained
            return float('inf'), len(self.X_test), {
                "loss": float('inf'),
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc_roc": 0.5,
                "wallet_address": self.wallet_address if self.wallet_address else "unknown"
            }
    
    def save_metrics_history(self, filepath: str = "client_metrics.json"):
        """Save metrics history to a file."""
        with open(filepath, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Saved metrics history to {filepath}")


def start_client(
    server_address: str = "127.0.0.1:8080",
    ipfs_url: str = "http://127.0.0.1:5001",
    ganache_url: str = "http://127.0.0.1:7545",
    contract_address: Optional[str] = None,
    wallet_address: Optional[str] = None,
    private_key: Optional[str] = None,
    client_id: Optional[str] = None,
    ensemble_size: int = 8,
    ga_generations: int = 3,
    ga_population_size: int = 8,
    train_file: Optional[str] = None,
    test_file: Optional[str] = None
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
        train_file: Path to the training data file
        test_file: Path to the test data file
    """
    # Extract client ID from training file path if not explicitly provided
    if client_id is None and train_file is not None:
        # Try to extract client ID from the filename
        filename = os.path.basename(train_file)
        if filename.startswith("client-") and "_train" in filename:
            # Extract client-X from filename like "client-1_train.txt"
            client_id = filename.split("_train")[0]
            logger.info(f"Extracted client ID from training file: {client_id}")
        else:
            # Use a default client ID with timestamp to avoid collisions
            timestamp = int(time.time())
            client_id = f"client-{timestamp}"
            logger.info(f"Using generated client ID: {client_id}")
    elif client_id is None:
        # Default client ID if no train file or explicit ID provided
        client_id = f"client-{os.getpid()}"
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
    
    # Determine dataset files to load
    if train_file is None or test_file is None:
        # If files aren't explicitly provided, derive from client ID
        derived_train_file = f"{client_id}_train.txt"
        derived_test_file = f"{client_id}_test.txt"
        
        # Check if derived files exist
        if not os.path.exists(derived_train_file) or not os.path.exists(derived_test_file):
            logger.warning(f"Derived dataset files not found: {derived_train_file}, {derived_test_file}")
            
            # Try to find in data directory
            data_dir_train = f"data/{client_id}_train.txt"
            data_dir_test = f"data/{client_id}_test.txt"
            
            if os.path.exists(data_dir_train) and os.path.exists(data_dir_test):
                logger.info(f"Found dataset files in data directory")
                train_file = data_dir_train
                test_file = data_dir_test
            else:
                # Fall back to default client-1 files
                logger.warning(f"Using default dataset files")
                train_file = "client-1_train.txt"
                test_file = "client-1_test.txt"
                
                # Check if default files exist in data directory
                if not os.path.exists(train_file) or not os.path.exists(test_file):
                    data_dir_default_train = "data/client-1_train.txt"
                    data_dir_default_test = "data/client-1_test.txt"
                    
                    if os.path.exists(data_dir_default_train) and os.path.exists(data_dir_default_test):
                        train_file = data_dir_default_train
                        test_file = data_dir_default_test
        else:
            train_file = derived_train_file
            test_file = derived_test_file
    
    logger.info(f"Using dataset files: {train_file} and {test_file}")
    
    # Create client
    client = GAStackingClient(
        X_train=X_train,
        y_train=y_train,
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
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8088", help="Server address (host:port)")
    parser.add_argument("--ipfs-url", type=str, default="http://127.0.0.1:5001/api/v0", help="IPFS API URL")
    parser.add_argument("--ganache-url", type=str, default="http://192.168.1.146:7545", help="Ganache blockchain URL")
    parser.add_argument("--contract-address", type=str, help="Address of deployed EnhancedModelRegistry contract")
    parser.add_argument("--wallet-address", type=str, help="Client's Ethereum wallet address")
    parser.add_argument("--private-key", type=str, help="Client's private key (for signing transactions)")
    parser.add_argument("--client-id", type=str, help="Client identifier")
    parser.add_argument("--ensemble-size", type=int, default=8, help="Number of models in the ensemble")
    parser.add_argument("--ga-generations", type=int, default=3, help="Number of GA generations to run")
    parser.add_argument("--ga-population-size", type=int, default=10, help="Size of GA population")
    parser.add_argument("--train-file", type=str, help="Path to training data file")
    parser.add_argument("--test-file", type=str, help="Path to test data file")
    args = parser.parse_args()
    
    # Check if contract address is stored in file
    if args.contract_address is None:
        try:
            with open("contract_address.txt", "r") as f:
                args.contract_address = f.read().strip()
                print(f"Loaded contract address from file: {args.contract_address}")
        except FileNotFoundError:
            print("No contract address provided or found in file")
    
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
        train_file=args.train_file,
        test_file=args.test_file
    )