"""
Enhanced Federated Learning Client with GA-Stacking Ensemble optimization.
Extends the base client with local ensemble optimization capabilities.
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

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch.utils.data
from torch.utils.data import DataLoader, Dataset

import random
import threading
from web3.exceptions import TransactionNotFound
from hexbytes import HexBytes
import pandas as pd

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
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

from ipfs_connector import IPFSConnector
from blockchain_connector import BlockchainConnector
from ga_stacking import (
    GAStacking, EnsembleModel, BaseModelWrapper, MetaLearnerWrapper,
    create_base_models, get_ensemble_state_dict, load_ensemble_from_state_dict
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("FL-Client-Ensemble")


class GAStackingClient(fl.client.NumPyClient):
    """Federated Learning client with GA-Stacking ensemble optimization."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        ensemble_size: int = 5,
        ipfs_connector: Optional[IPFSConnector] = None,
        blockchain_connector: Optional[BlockchainConnector] = None,
        wallet_address: Optional[str] = None,
        private_key: Optional[str] = None,
        device: str = "cpu",
        client_id: str = None,
        ga_generations: int = 3,
        ga_population_size: int = 8
    ):
        """
        Initialize GA-Stacking client.
        
        Args:
            input_dim: Input dimension for models
            output_dim: Output dimension for models
            train_loader: Training data loader
            test_loader: Test data loader
            ensemble_size: Number of models in the ensemble
            ipfs_connector: IPFS connector
            blockchain_connector: Blockchain connector
            wallet_address: Client's wallet address
            private_key: Client's private key
            device: Device to use for computation
            client_id: Client identifier
            ga_generations: Number of GA generations to run
            ga_population_size: Size of GA population
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.ensemble_size = ensemble_size
        self.ipfs = ipfs_connector or IPFSConnector()
        self.blockchain = blockchain_connector
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.device = torch.device(device)
        self.client_id = client_id or f"client-{os.getpid()}"
        self.ga_generations = ga_generations
        self.ga_population_size = ga_population_size
        
        # Initialize ensemble models
        self.base_models, self.model_names = create_base_models(
            input_dim=input_dim,
            output_dim=output_dim
        )
        
        # Initialize GA-Stacking optimizer
        self.ga_stacking = None
        self.ensemble_model = None
        
        # Metrics storage
        self.metrics_history = []
        
        logger.info(f"Initialized {self.client_id} with {ensemble_size} base models")
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
    
    def split_train_val(self, validation_split: float = 0.2):
        """
        Split the training data into training and validation sets.
        
        Args:
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Get the full dataset from the data loader
        train_dataset = self.train_loader.dataset
        
        # Calculate sizes
        val_size = int(len(train_dataset) * validation_split)
        train_size = len(train_dataset) - val_size
        
        # Split the dataset
        train_subset, val_subset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create new data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.train_loader.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.train_loader.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """
        Get model parameters as a list of NumPy arrays.
        
        For ensemble models, we flatten the parameters of all models and their weights.
        The server will need to know how to reconstruct this.
        """
        if self.ensemble_model is None:
            # If ensemble not trained yet, use the first base model
            return [val.cpu().numpy() for _, val in self.base_models[0].state_dict().items()]
        
        # Get ensemble state including model weights and ensemble weights
        ensemble_state = get_ensemble_state_dict(self.ensemble_model)
        
        # Serialize ensemble state to JSON and convert to ndarray
        ensemble_bytes = json.dumps(ensemble_state).encode('utf-8')
        return [np.frombuffer(ensemble_bytes, dtype=np.uint8)]
    
    def set_parameters_from_ensemble_state(self, ensemble_state: Dict[str, Any]) -> None:
        """
        Set ensemble model parameters from an ensemble state dictionary.
        
        Args:
            ensemble_state: Ensemble state dictionary with weights and model states
        """
        try:
            # Check if model classes and params match current models
            if len(ensemble_state.get("model_state_dicts", [])) != len(self.base_models):
                logger.warning(
                    f"Ensemble state has {len(ensemble_state.get('model_state_dicts', []))} models, "
                    f"but we have {len(self.base_models)} base models. Using subset of models."
                )
            
            # Get model names
            model_names = ensemble_state.get("model_names", [f"model_{i}" for i in range(len(self.base_models))])
            
            # Identify base models and meta models
            base_indices = []
            meta_indices = []
            
            for i, name in enumerate(model_names):
                if i >= len(self.base_models):
                    break
                    
                if name == "meta_lr" or "MetaLearner" in name:
                    meta_indices.append(i)
                else:
                    base_indices.append(i)
            
            # Load weights for each model
            for i, model in enumerate(self.base_models):
                if i >= len(ensemble_state.get("model_state_dicts", [])):
                    continue
                    
                # Get state dict for this model
                state_dict = ensemble_state["model_state_dicts"][i]
                
                # Special handling for sklearn wrappers
                if hasattr(model, 'set_parameters'):
                    # For scikit-learn wrapped models
                    try:
                        # Make a copy of the state dict to avoid modifying the original
                        modified_dict = state_dict.copy()
                        
                        # If this is a meta learner, ensure input_dim is correct
                        if i in meta_indices:
                            modified_dict["input_dim"] = len(base_indices)
                            
                        model.set_parameters(modified_dict)
                    except Exception as e:
                        logger.error(f"Error setting parameters for model {i}: {e}")
                else:
                    # For PyTorch models
                    try:
                        # Convert state dict values to tensors
                        tensor_state_dict = {}
                        for key, value in state_dict.items():
                            if key not in ["model_type", "estimator", "input_dim", "output_dim"]:
                                try:
                                    # Handle different types of values
                                    if isinstance(value, (list, np.ndarray)):
                                        tensor_state_dict[key] = torch.tensor(value, device=self.device)
                                    elif isinstance(value, dict):
                                        # Handle nested dictionaries (unlikely but possible)
                                        tensor_state_dict[key] = {
                                            k: torch.tensor(v, device=self.device) if isinstance(v, (list, np.ndarray)) else v
                                            for k, v in value.items()
                                        }
                                    elif isinstance(value, str):
                                        # Skip string values (metadata)
                                        continue
                                    else:
                                        # Handle scalar values
                                        tensor_state_dict[key] = torch.tensor([value], device=self.device)
                                except Exception as tensor_e:
                                    logger.error(f"Error converting to tensor for key {key}: {tensor_e}")
                                    # Skip this key
                                    continue
                        
                        # Check if we have any tensor values
                        if tensor_state_dict:
                            # Update model parameters
                            model.load_state_dict(tensor_state_dict)
                    except Exception as e:
                        logger.error(f"Error loading state dict for model {i}: {e}")
            
            # Create ensemble model with loaded weights
            weights = ensemble_state.get("weights", [1.0/len(self.base_models)] * len(self.base_models))
            
            # Ensure weights match the number of models
            if len(weights) != len(self.base_models):
                logger.warning(f"Weight count mismatch. Using equal weights.")
                weights = [1.0/len(self.base_models)] * len(self.base_models)
            
            # Create the ensemble model
            self.ensemble_model = EnsembleModel(
                models=self.base_models,
                weights=weights,
                model_names=model_names[:len(self.base_models)],
                device=self.device
            )
            self.check_and_fix_zero_coefficients()
            logger.info(f"Ensemble model updated with {len(self.base_models)} models")
            
        except Exception as e:
            logger.error(f"Failed to set parameters from ensemble state: {e}")
            # No fallback needed here since we catch exceptions for each model individually
            weights = [1.0/len(self.base_models)] * len(self.base_models)
            self.ensemble_model = EnsembleModel(
                models=self.base_models,
                weights=weights,
                model_names=self.model_names,
                device=self.device
            )
    
    def set_parameters_individual(self, parameters: List[np.ndarray]) -> None:
        """
        Set individual model parameters from a list of NumPy arrays.
        
        This is used as a fallback when ensemble state loading fails.
        
        Args:
            parameters: List of parameter arrays
        """
        if len(parameters) == 1 and parameters[0].dtype == np.uint8:
            # Try to interpret as serialized ensemble state
            try:
                ensemble_bytes = parameters[0].tobytes()
                ensemble_state = json.loads(ensemble_bytes.decode('utf-8'))
                self.set_parameters_from_ensemble_state(ensemble_state)
                return
            except Exception as e:
                logger.error(f"Failed to deserialize ensemble state: {e}")
                logger.warning("Falling back to individual model parameter loading")
        
        # Apply parameters to the first model only (fallback)
        try:
            params_dict = zip(self.base_models[0].state_dict().keys(), parameters)
            state_dict = {k: torch.Tensor(v) for k, v in params_dict}
            self.base_models[0].load_state_dict(state_dict, strict=True)
            logger.warning("Applied parameters to first base model only (fallback mode)")
        except Exception as e:
            logger.error(f"Failed to set individual parameters: {e}")
    
    def _create_models_from_server_configs(self, configs, input_dim, output_dim, device):
        """Create models specifically matched to server config types."""
        models = []
        model_names = []
        
        for config in configs:
            model_type = config.get("estimator", "")
            
            # Ensure dimensions are correct
            model_input_dim = config.get("input_dim", input_dim)
            model_output_dim = config.get("output_dim", output_dim) 
            
            # Create the appropriate model type
            if model_type == "lr":
                from sklearn.linear_model import LogisticRegression
                model = BaseModelWrapper("lr", LogisticRegression(solver='liblinear'), model_input_dim, model_output_dim)
            elif model_type == "svc":
                from sklearn.svm import SVC
                model = BaseModelWrapper("svc", SVC(probability=True), model_input_dim, model_output_dim)
            elif model_type == "rf":
                from sklearn.ensemble import RandomForestClassifier
                model = BaseModelWrapper("rf", RandomForestClassifier(n_estimators=100), model_input_dim, model_output_dim)
            elif model_type == "knn":
                from sklearn.neighbors import KNeighborsClassifier
                model = BaseModelWrapper("knn", KNeighborsClassifier(n_neighbors=5), model_input_dim, model_output_dim)
            elif model_type == "xgb":
                try:
                    from xgboost import XGBClassifier
                    model = BaseModelWrapper("xgb", XGBClassifier(use_label_encoder=False, eval_metric='logloss'), model_input_dim, model_output_dim)
                except ImportError:
                    # Fall back to RandomForest if XGBoost not available
                    from sklearn.ensemble import RandomForestClassifier
                    model = BaseModelWrapper("rf", RandomForestClassifier(n_estimators=100), model_input_dim, model_output_dim)
            elif model_type == "lgbm":
                try:
                    from lightgbm import LGBMClassifier
                    model = BaseModelWrapper("lgbm", LGBMClassifier(), model_input_dim, model_output_dim)
                except ImportError:
                    # Fall back to RandomForest if LightGBM not available
                    from sklearn.ensemble import RandomForestClassifier
                    model = BaseModelWrapper("rf", RandomForestClassifier(n_estimators=100), model_input_dim, model_output_dim)
            elif model_type == "catboost":
                try:
                    from catboost import CatBoostClassifier
                    model = BaseModelWrapper("catboost", CatBoostClassifier(verbose=0), model_input_dim, model_output_dim)
                except ImportError:
                    # Fall back to RandomForest if CatBoost not available
                    from sklearn.ensemble import RandomForestClassifier
                    model = BaseModelWrapper("rf", RandomForestClassifier(n_estimators=100), model_input_dim, model_output_dim)
            elif model_type == "meta_lr":
                # Meta-learner has special input dimension handling
                model = MetaLearnerWrapper(model_input_dim, model_output_dim)
            else:
                # Default fallback
                from sklearn.linear_model import LogisticRegression
                model = BaseModelWrapper("lr", LogisticRegression(solver='liblinear'), model_input_dim, model_output_dim)
            
            # Set model parameters if available
            if "model_params" in config and hasattr(model, 'set_params'):
                try:
                    model.set_params(config)
                except Exception as e:
                    logger.warning(f"Error setting parameters for {model_type}: {e}")
            
            # Add to lists
            models.append(model)
            model_names.append(model_type)
            
        return models, model_names
    
    def set_parameters_from_ipfs(self, ipfs_hash: str) -> None:
        """Set model parameters from IPFS with improved model type handling."""
        try:
            # Get model data from IPFS
            model_data = self.ipfs.get_json(ipfs_hash)
            
            if model_data and "ensemble_state" in model_data:
                # Load ensemble state
                ensemble_state = model_data["ensemble_state"]
                
                # Check for model parameters in ensemble state
                if "model_parameters" in ensemble_state:
                    configs = ensemble_state["model_parameters"]
                    logger.info(f"Found {len(configs)} model configurations in ensemble state")
                    
                    # First identify which models are base models vs meta learners
                    base_configs = []
                    meta_configs = []
                    
                    for config in configs:
                        if config.get("estimator") == "meta_lr" or config.get("model_type") == "meta_lr":
                            meta_configs.append(config)
                        else:
                            base_configs.append(config)
                    
                    # Process base model configs first
                    for config in base_configs:
                        # Ensure each config has correct input/output dimensions
                        if "input_dim" not in config:
                            config["input_dim"] = self.input_dim
                        if "output_dim" not in config:
                            config["output_dim"] = self.output_dim
                    
                    # Then process meta learner configs
                    for config in meta_configs:
                        # CRITICAL FIX: Meta learner should have input dim equal to number of base models
                        config["input_dim"] = len(base_configs)
                        config["meta_input_dim"] = len(base_configs)  # For backward compatibility
                        config["expected_input_dim"] = len(base_configs)  # For our improved MetaLearnerWrapper
                        if "output_dim" not in config:
                            config["output_dim"] = self.output_dim
                    
                    # Combine all configs again
                    all_configs = base_configs + meta_configs
                    
                    # Initialize models using specialized function for server configs
                    self.base_models, self.model_names = self._create_models_from_server_configs(
                        all_configs,
                        input_dim=self.input_dim,
                        output_dim=self.output_dim,
                        device=self.device
                    )
                    
                    # Create ensemble model with weights
                    weights = ensemble_state.get("weights", [1.0/len(self.base_models)] * len(self.base_models))
                    
                    # Ensure weights match number of models
                    if len(weights) != len(self.base_models):
                        logger.warning(f"Weight count mismatch. Expected {len(self.base_models)}, got {len(weights)}")
                        weights = [1.0/len(self.base_models)] * len(self.base_models)
                    
                    self.ensemble_model = EnsembleModel(
                        models=self.base_models,
                        weights=weights,
                        model_names=self.model_names,
                        device=self.device
                    )
                    
                    logger.info(f"Ensemble model loaded from IPFS: {ipfs_hash}")
                
            elif model_data and "state_dict" in model_data:
                # Legacy format - single model
                state_dict = {
                    k: torch.tensor(v, device=self.device)
                    for k, v in model_data["state_dict"].items()
                }
                
                # Try to update the first base model
                try:
                    self.base_models[0].load_state_dict(state_dict)
                    logger.info(f"First base model loaded from IPFS: {ipfs_hash}")
                except Exception as e:
                    logger.warning(f"Failed to load state dict into base model: {e}")
                    
            else:
                logger.error(f"Invalid model data from IPFS: {ipfs_hash}")
                
        except Exception as e:
            logger.error(f"Failed to load model from IPFS: {e}")
            logger.error(f"Error details: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def save_ensemble_to_ipfs(
        self, round_num: int, metrics: Dict[str, Scalar] = None
    ) -> str:
        """
        Save ensemble model to IPFS.
        
        Args:
            round_num: Current round number
            metrics: Optional metrics to include
            
        Returns:
            IPFS hash of the saved model
        """
        if self.ensemble_model is None:
            logger.warning("No ensemble model to save")
            return None
        
        # Get ensemble state
        ensemble_state = get_ensemble_state_dict(self.ensemble_model)
        
        # Create metadata
        model_metadata = {
            "ensemble_state": ensemble_state,
            "info": {
                "round": round_num,
                "client_id": self.client_id,
                "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ensemble_size": len(self.ensemble_model.models),
                "weights": ensemble_state["weights"]
            }
        }
        
        # Add metrics if provided
        if metrics:
            model_metadata["info"]["metrics"] = metrics
        
        # Store in IPFS
        ipfs_hash = self.ipfs.add_json(model_metadata)
        logger.info(f"Stored ensemble model in IPFS: {ipfs_hash}")
        
        return ipfs_hash
    
    def train_ensemble(self, validation_split: float = 0.2):
        """Train the ensemble using GA-Stacking with proper meta-learner handling."""
        # Split data into train and validation
        train_loader, val_loader = self.split_train_val(validation_split)
        
        # Separate models
        base_models = []
        meta_models = []
        base_model_names = []
        meta_model_names = []
        
        for i, (model, name) in enumerate(zip(self.base_models, self.model_names)):
            if name == "meta_lr" or "MetaLearner" in name or "meta" in name.lower():
                meta_models.append(model)
                meta_model_names.append(name)
            else:
                base_models.append(model)
                base_model_names.append(name)
        
        # Check and fix zero coefficients before training
        self.check_and_fix_zero_coefficients()

        # 1. First train all base models
        for i, (model, name) in enumerate(zip(base_models, base_model_names)):
            try:
                # Skip if the model is already trained
                if hasattr(model, 'is_initialized') and model.is_initialized:
                    logger.info(f"Skipping training for {name} (already trained)")
                else:
                    self._train_single_model(model, train_loader, epochs=5)
                    logger.info(f"Base model {i+1}/{len(base_models)} ({name}) trained")
            except Exception as e:
                logger.error(f"Error training base model {name}: {e}")
        
        # 2. Collect validation data for meta-learner training
        X_val_all = []
        y_val_all = []
        
        for batch_idx, (data, target) in enumerate(val_loader):
            X_val_all.append(data.to(self.device))
            y_val_all.append(target.to(self.device))
        
        # Concatenate batches
        if X_val_all and y_val_all:
            X_val = torch.cat(X_val_all)
            y_val = torch.cat(y_val_all)
        else:
            logger.error("No validation data available for meta-learner training")
            X_val, y_val = None, None
        
        # 3. Generate base model predictions for meta-learner training
        if X_val is not None:
            base_model_preds = []
            
            for model, name in zip(base_models, base_model_names):
                try:
                    with torch.no_grad():
                        model.eval()
                        preds = model(X_val)
                        if isinstance(preds, torch.Tensor):
                            preds = preds.cpu().detach().numpy()
                        else:
                            preds = np.array(preds)
                        
                        # Ensure 2D shape
                        if len(preds.shape) == 1:
                            preds = preds.reshape(-1, 1)
                        
                        base_model_preds.append(preds)
                        logger.info(f"Generated predictions from {name} for meta-learner training")
                except Exception as e:
                    logger.error(f"Error generating predictions from {name}: {e}")
                    # Add zeros as fallback
                    base_model_preds.append(np.zeros((len(X_val), 1)))
            
            # 4. CRITICAL FIX: TRAIN META-LEARNERS WITH COMBINED BASE MODEL PREDICTIONS
            if base_model_preds:
                # Stack all base model predictions side by side
                meta_features = np.hstack(base_model_preds)
                
                for meta_model, meta_name in zip(meta_models, meta_model_names):
                    try:
                        # THIS IS THE KEY FIX - update meta_model.input_dim to match base_model_preds
                        meta_model.input_dim = meta_features.shape[1]
                        
                        # If it has a linear layer, update that too
                        if hasattr(meta_model, 'linear'):
                            meta_model.linear = nn.Linear(meta_features.shape[1], meta_model.output_dim, 
                                                        device=self.device).to(dtype=torch.float32)
                        
                        # Convert targets to numpy for sklearn models
                        if isinstance(y_val, torch.Tensor):
                            y_meta = y_val.cpu().numpy()
                        else:
                            y_meta = y_val
                        
                        # Flatten if needed
                        if len(y_meta.shape) > 1 and y_meta.shape[1] == 1:
                            y_meta = y_meta.ravel()
                        
                        # Train using appropriate method
                        if hasattr(meta_model, 'model') and hasattr(meta_model.model, 'fit'):
                            meta_model.model.fit(meta_features, y_meta)
                            
                            # Update PyTorch layer with sklearn coefficients
                            if hasattr(meta_model, 'linear') and hasattr(meta_model.model, 'coef_'):
                                meta_model.linear.weight.data = torch.tensor(
                                    meta_model.model.coef_, device=self.device, dtype=torch.float32)
                                meta_model.linear.bias.data = torch.tensor(
                                    meta_model.model.intercept_, device=self.device, dtype=torch.float32)
                                
                            meta_model.is_initialized = True
                            logger.info(f"Meta-learner {meta_name} trained on base model predictions " 
                                    f"with input dim {meta_features.shape[1]}")
                        else:
                            # Create PyTorch DataLoader for PyTorch models
                            meta_dataset = torch.utils.data.TensorDataset(
                                torch.tensor(meta_features, dtype=torch.float32), 
                                torch.tensor(y_meta, dtype=torch.float32)
                            )
                            meta_loader = torch.utils.data.DataLoader(
                                meta_dataset, batch_size=32, shuffle=True)
                            
                            self._train_single_model(meta_model, meta_loader, epochs=10)
                    except Exception as e:
                        logger.error(f"Error training meta-learner {meta_name}: {e}")
        
        # 5. Run GA-Stacking with fixed model dimensions
        # START THE GA-STACKING WITH PROPERLY INITIALIZED MODELS
        all_models = base_models + meta_models
        all_model_names = base_model_names + meta_model_names
        
        # Initialize GA-Stacking
        self.ga_stacking = GAStacking(
            base_models=all_models,
            model_names=all_model_names,
            population_size=self.ga_population_size,
            generations=self.ga_generations,
            device=self.device
        )
        
        # Run optimization
        logger.info("Starting GA-Stacking optimization")
        try:
            best_weights = self.ga_stacking.optimize(train_loader, val_loader)
            
            # Get optimized ensemble
            self.ensemble_model = self.ga_stacking.get_ensemble_model()
            
            # Log final weights
            weight_str = ", ".join([f"{name}: {weight:.4f}" for name, weight in zip(all_model_names, best_weights)])
            logger.info(f"GA-Stacking complete. Ensemble weights: {weight_str}")
            
            return self.ensemble_model
        except Exception as e:
            logger.error(f"Error in GA-Stacking optimization: {e}")
            # Fall back to equal weighting
            weights = np.ones(len(all_models)) / len(all_models)
            self.ensemble_model = EnsembleModel(
                models=all_models,
                weights=weights,
                model_names=all_model_names,
                device=self.device
            )
            logger.warning("Using equal weights due to GA-Stacking failure")
            return self.ensemble_model
    
    def _check_model_weights(self, model):
        """Kiểm tra và in thông tin trọng số mô hình để debug."""
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'coef_'):
                coef = model.model.coef_
                logger.info(f"Model coefficients - shape: {coef.shape}, mean: {np.mean(coef):.4f}, min: {np.min(coef):.4f}, max: {np.max(coef):.4f}")
                if np.all(coef == 0):
                    logger.warning("All coefficients are zero!")
                    
            elif hasattr(model, 'linear') and hasattr(model.linear, 'weight'):
                weight = model.linear.weight.data.cpu().numpy()
                logger.info(f"Linear weights - shape: {weight.shape}, mean: {np.mean(weight):.4f}, min: {np.min(weight):.4f}, max: {np.max(weight):.4f}")
                if np.all(weight == 0):
                    logger.warning("All weights are zero!")
        except Exception as e:
            logger.error(f"Error checking model weights: {e}")
    
    def _train_single_model(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        epochs: int = 5,    
        learning_rate: float = 0.01
    ) -> None:
        """
        Huấn luyện mô hình đơn lẻ với cải thiện cho dữ liệu mất cân bằng.
        """
        # Skip training for scikit-learn model wrappers with direct fit method
        if hasattr(model, "fit"):
            try:
                # Collect all data from loader
                X_all = []
                y_all = []
                for data, target in train_loader:
                    X_all.append(data.cpu().numpy())
                    y_all.append(target.cpu().numpy())
                    
                if not X_all or not y_all:
                    logger.warning("No training data available")
                    return
                    
                X = np.vstack(X_all)
                y = np.vstack(y_all).flatten()
                
                # Fit model directly
                model.fit(X, y)
                logger.info(f"Trained model with {len(X)} samples")
                return
            except Exception as e:
                logger.error(f"Error in direct model fitting: {e}")
        
        # For PyTorch models or models without direct fit
        model.train()
        
        # Sử dụng BCE với positive weight thay vì MSE
        # Tính toán positive weight dựa trên tỷ lệ mẫu dương/âm
        pos_samples = sum(batch[1].sum().item() for batch in train_loader)
        total_samples = sum(len(batch[1]) for batch in train_loader)
        neg_samples = total_samples - pos_samples
        
        # Tính pos_weight: tỷ lệ mẫu âm tính / mẫu dương tính
        pos_weight = neg_samples / max(1, pos_samples)  # Tránh chia cho 0
        logger.info(f"Class imbalance - positive samples: {pos_samples}/{total_samples} ({pos_samples/total_samples*100:.2f}%)")
        logger.info(f"Using positive weight: {pos_weight:.2f} for BCE loss")
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=self.device))
        
        # Check if model has parameters
        if sum(p.numel() for p in model.parameters() if p.requires_grad) == 0:
            logger.warning(f"Model has no trainable parameters, skipping training")
            return
            
        # Sử dụng Adam thay vì SGD
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                
                # Adjust shapes if needed
                if output.shape != target.shape:
                    if len(output.shape) == 1:
                        output = output.unsqueeze(1)
                    if len(target.shape) == 1:
                        target = target.unsqueeze(1)
                
                # Tính loss với BCE
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(data)
                epoch_samples += len(data)
            
            avg_epoch_loss = epoch_loss / epoch_samples
            if epoch % 2 == 0:
                logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
    
    def _initialize_models_from_server_config(self, configs):
        """Initialize models from server configuration with proper type handling."""
        models = []
        model_names = []
        
        for config in configs:
            model_type = config.get("estimator", "")
            
            if model_type == "lr":
                model = LinearRegressionWrapper(self.input_dim, self.output_dim)
                models.append(model)
                model_names.append(model_type)
                
            elif model_type == "svc":
                model = SVCWrapper(self.input_dim, self.output_dim)
                models.append(model)
                model_names.append(model_type)
                
            elif model_type == "rf":
                model = RandomForestWrapper(self.input_dim, self.output_dim)
                models.append(model)
                model_names.append(model_type)
                
            elif model_type == "xgb":
                # Fallback to GBM if XGBoost wrapper not implemented
                model = GradientBoostingWrapper(self.input_dim, self.output_dim)
                models.append(model)
                model_names.append(model_type)
                
            elif model_type == "lgbm":
                # Fallback to GBM if LightGBM wrapper not implemented
                model = GradientBoostingWrapper(self.input_dim, self.output_dim)
                models.append(model)
                model_names.append(model_type)
                
            elif model_type == "catboost":
                # Fallback to GBM if CatBoost wrapper not implemented
                model = GradientBoostingWrapper(self.input_dim, self.output_dim)
                models.append(model)
                model_names.append(model_type)
                
            elif model_type == "knn":
                # Use a simple wrapper or fallback to linear
                model = LinearRegressionWrapper(self.input_dim, self.output_dim)
                models.append(model)
                model_names.append(model_type)
                
            elif model_type == "meta_lr":
                # Meta learner needs input_dim = number of base models
                model = MetaLearnerWrapper(len([c for c in configs if c.get("estimator") != "meta_lr"]), self.output_dim)
                models.append(model)
                model_names.append(model_type)
                
            else:
                # Default to linear model
                model = LinearModel(self.input_dim, self.output_dim)
                models.append(model)
                model_names.append("linear")
        
        # Move all models to device
        for model in models:
            model.to(self.device)
        
        return models, model_names
    
    def initialize_models_from_config(
        self, 
        config: Dict[str, Any]
    ) -> Tuple[List[nn.Module], List[str]]:
        """
        Initialize models from configuration provided by the server.
        
        Args:
            config: Configuration from the server that may include initial_ensemble
            
        Returns:
            Tuple of (list of models, list of model names)
        """
        initial_ensemble = config.get("initial_ensemble", None)
        
        if initial_ensemble:
            # Check if we have model_parameters or the old format
            if "model_parameters" in initial_ensemble:
                logger.info(f"Initializing models from server-provided model_parameters ({len(initial_ensemble['model_parameters'])} models)")
                return self._initialize_models_from_server_config(initial_ensemble["model_parameters"])
            else:
                logger.info(f"Initializing models from server-provided configuration ({len(initial_ensemble)} models)")
                models, model_names = create_model_ensemble_from_config(
                    initial_ensemble,
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    device=self.device
                )
                
                # Log the model types received
                model_types = ", ".join(model_names)
                logger.info(f"Initialized models: {model_types}")
                
                return models, model_names
        else:
            # Fall back to default model creation
            logger.info(f"No initial configuration provided, creating default ensemble with {self.ensemble_size} models")
            return create_model_ensemble(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                ensemble_size=self.ensemble_size,
                device=self.device
            )

    def fit(self, parameters: Parameters, config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train the model on the local dataset.

        For GA-Stacking, we train individual models and then optimize the ensemble.
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

        # Get global model from IPFS if hash is provided
        ipfs_hash = config.get("ipfs_hash", None)
        round_num = config.get("server_round", 0)

        # If this is the first round, check for initial ensemble configuration
        if round_num == 1 and "initial_ensemble" in config:
            # Initialize models from configuration
            initial_config = config.get("initial_ensemble", {}).get("model_parameters", [])
            self.base_models, self.model_names = self._initialize_models_from_server_config(initial_config)
            logger.info(f"Initialized {len(self.base_models)} models from server configuration")
        elif ipfs_hash:
            # Load model from IPFS with dimension fix
            try:
                self.set_parameters_from_ipfs(ipfs_hash)
                logger.info(f"Model loaded from IPFS: {ipfs_hash}")
            except Exception as e:
                logger.error(f"Failed to load model from IPFS: {str(e)}")
                # Continue with current models
        else:
            # Fallback to direct parameters
            params_arrays = parameters_to_ndarrays(parameters)
            self.set_parameters_individual(params_arrays)

        # Get training config
        local_epochs = int(config.get("local_epochs", 5))
        do_ga_stacking = bool(config.get("ga_stacking", True))
        validation_split = float(config.get("validation_split", 0.2))

        # Verify and fix models before training
        self._verify_and_fix_models()

        # Add this line to check and fix zero coefficients
        self.check_and_fix_zero_coefficients()
        
        # Create validation set for GA-Stacking optimization
        train_data, val_data = self._create_validation_split(validation_split)
        
        # Store generation history if GA-Stacking is used
        ga_convergence_data = None

        # Run GA-Stacking to optimize ensemble weights
        if do_ga_stacking:
            logger.info("Performing GA-Stacking optimization")
            try:
                # Modified to capture GA convergence metrics
                self.train_ensemble(validation_split=validation_split)
                
                # Capture GA convergence data for reward calculation
                ga_convergence_data = {
                    "generations": self.ga_optimizer.generations if hasattr(self, 'ga_optimizer') else [],
                    "convergence_rate": self._calculate_ga_convergence_rate() if hasattr(self, '_calculate_ga_convergence_rate') else 0.5
                }
            except Exception as e:
                logger.error(f"Error in GA-Stacking: {str(e)}")
                # Fall back to equal weights
                weights = np.ones(len(self.base_models)) / len(self.base_models)
                self.ensemble_model = EnsembleModel(
                    models=self.base_models,
                    weights=weights,
                    model_names=self.model_names,
                    device=self.device
                )
        else:
            logger.info("Skipping GA-Stacking, training individual models")
            # Train individual models
            for i, model in enumerate(self.base_models):
                try:
                    self._train_single_model(model, self.train_loader, epochs=local_epochs)
                    logger.info(f"Model {i+1}/{len(self.base_models)} trained")
                except Exception as e:
                    logger.error(f"Error training model {i+1}: {str(e)}")

            # Equal weighting if not using GA-Stacking
            weights = np.ones(len(self.base_models)) / len(self.base_models)
            self.ensemble_model = EnsembleModel(
                models=self.base_models,
                weights=weights,
                model_names=self.model_names,
                device=self.device
            )

        # Ensure we have an ensemble model
        if self.ensemble_model is None:
            logger.warning("No ensemble model created, creating one with equal weights")
            weights = np.ones(len(self.base_models)) / len(self.base_models)
            self.ensemble_model = EnsembleModel(
                models=self.base_models,
                weights=weights,
                model_names=self.model_names,
                device=self.device
            )

        # Calculate GA-Stacking specific metrics for reward system
        ga_stacking_metrics = self._calculate_ga_stacking_metrics(val_data, ga_convergence_data)

        # Evaluate the ensemble
        try:
            accuracy, loss, metrics = self._evaluate_ensemble(self.test_loader)
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {str(e)}")
            accuracy, loss = 0.0, float('inf')

        # Save ensemble to IPFS
        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            # Add fraud-specific metrics
            "precision": float(metrics['precision']),
            "recall": float(metrics['recall']),
            "f1_score": float(metrics['f1_score']),
            "auc_roc": float(metrics.get('auc_roc', 0.5)),
            # Add GA-Stacking specific metrics
            "ensemble_accuracy": float(ga_stacking_metrics.get("ensemble_accuracy", accuracy)),
            "diversity_score": float(ga_stacking_metrics.get("diversity_score", 0.0)),
            "generalization_score": float(ga_stacking_metrics.get("generalization_score", 0.0)),
            "convergence_rate": float(ga_stacking_metrics.get("convergence_rate", 0.5)),
            "avg_base_model_score": float(ga_stacking_metrics.get("avg_base_model_score", 0.0)),
        }

        try:
            client_ipfs_hash = self.save_ensemble_to_ipfs(round_num, metrics)
        except Exception as e:
            logger.error(f"Failed to save ensemble to IPFS: {str(e)}")
            client_ipfs_hash = None

        # Add to metrics history
        self.metrics_history.append({
            "round": round_num,
            "fit_loss": float(loss),
            "accuracy": float(accuracy),
            "ensemble_weights": self.ensemble_model.weights.cpu().numpy().tolist() if self.ensemble_model else [],
            "ipfs_hash": client_ipfs_hash,
            "wallet_address": self.wallet_address if self.wallet_address else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Add GA-Stacking metrics to history
            "ga_stacking_metrics": ga_stacking_metrics
        })

        # Include IPFS hash in metrics
        metrics["ipfs_hash"] = ipfs_hash  # Return the server's hash for tracking
        metrics["client_ipfs_hash"] = client_ipfs_hash
        metrics["wallet_address"] = self.wallet_address if self.wallet_address else "unknown"
        metrics["ensemble_size"] = len(self.base_models)
        
        # Add GA-Stacking specific metrics to the returned metrics
        for key, value in ga_stacking_metrics.items():
            metrics[key] = value

        # Get the ensemble parameters
        try:
            parameters_updated = self.get_parameters(config)
        except Exception as e:
            logger.error(f"Error getting parameters: {str(e)}")
            parameters_updated = parameters_to_ndarrays(parameters)  # Return original parameters

        # Get number of training samples
        num_samples = len(self.train_loader.dataset)

        # Make sure metrics only contains serializable values
        for key in list(metrics.keys()):
            if not isinstance(metrics[key], (int, float, str, bool, list)):
                metrics[key] = str(metrics[key])

        # Return the raw parameters (not wrapped in Flower Parameters)
        return parameters_updated, num_samples, metrics

    def _calculate_ga_stacking_metrics(self, validation_data, ga_convergence_data=None):
        """
        Calculate GA-Stacking specific metrics for reward calculation.
        
        Args:
            validation_data: Tuple of (X_val, y_val) for evaluation
            ga_convergence_data: Optional dictionary containing GA convergence info
            
        Returns:
            dict: GA-Stacking metrics
        """
        metrics = {}
    
        try:
            X_val, y_val = validation_data
            
            # Handle empty validation data
            if X_val.numel() == 0 or y_val.numel() == 0:
                logger.warning("Empty validation data. Using default metrics.")
                return {
                    "ensemble_accuracy": 0.0,
                    "diversity_score": 0.0,
                    "generalization_score": 0.0,
                    "convergence_rate": 0.5,
                    "avg_base_model_score": 0.0,
                    "final_score": 0
                }
            
            # 1. Calculate ensemble accuracy on validation data
            if self.ensemble_model is None:
                logger.warning("No ensemble model for metrics calculation. Creating one with equal weights.")
                weights = np.ones(len(self.base_models)) / len(self.base_models)
                self.ensemble_model = EnsembleModel(
                    models=self.base_models,
                    weights=weights,
                    model_names=self.model_names,
                    device=self.device
                )
            
            # Ensure data is float32 for consistent dtype
            X_val = X_val.to(dtype=torch.float32, device=self.device)
            y_val = y_val.to(dtype=torch.float32, device=self.device)
                
            # Get ensemble predictions
            with torch.no_grad():
                ensemble_outputs = self.ensemble_model(X_val)
                
                # For regression tasks, we need to define what "accuracy" means
                # Here, we'll consider a prediction "correct" if it's within 0.5 of the true value
                threshold = 0.1
                ensemble_outputs = ensemble_outputs.to(dtype=torch.float32)
                y_val = y_val.to(dtype=torch.float32)
                
                # Ensure dimensions match
                if ensemble_outputs.shape != y_val.shape:
                    if len(ensemble_outputs.shape) == 1:
                        ensemble_outputs = ensemble_outputs.unsqueeze(1)
                    if len(y_val.shape) == 1:
                        y_val = y_val.unsqueeze(1)
                        
                correct = torch.sum(torch.abs(ensemble_outputs - y_val) < threshold).item()
                total = y_val.size(0)
                ensemble_accuracy = correct / total if total > 0 else 0.0
                    
            metrics["ensemble_accuracy"] = float(ensemble_accuracy)
            
            # 2. Calculate diversity among base models (include only base models, not meta)
            base_models = []
            base_model_names = []
            
            for i, (model, name) in enumerate(zip(self.base_models, self.model_names)):
                if name != "meta_lr" and "meta" not in name.lower():
                    base_models.append(model)
                    base_model_names.append(name)
            
            diversity_score = self._calculate_model_diversity(validation_data, base_models, base_model_names)
            metrics["diversity_score"] = float(diversity_score)
            
            # 3. Calculate generalization score
            # We need to calculate on training data as well
            try:
                # Try to get a sample of training data
                train_sample_size = min(1000, X_val.size(0) * 4)  # Use at most 1000 samples or 4x validation size
                
                # Get random training samples
                train_indices = torch.randperm(len(self.train_loader.dataset))[:train_sample_size]
                train_data = []
                train_targets = []
                
                # Extract samples from dataset
                for idx in train_indices:
                    data, target = self.train_loader.dataset[idx]
                    train_data.append(data.unsqueeze(0))
                    train_targets.append(target.unsqueeze(0))
                    
                # Combine into tensors
                train_data_tensor = torch.cat(train_data, dim=0) if train_data else torch.tensor([])
                train_targets_tensor = torch.cat(train_targets, dim=0) if train_targets else torch.tensor([])
                
                if train_data_tensor.numel() > 0:
                    # Calculate training accuracy
                    with torch.no_grad():
                        train_outputs = self.ensemble_model(train_data_tensor)
                        correct = torch.sum(torch.abs(train_outputs - train_targets_tensor) < threshold).item()
                        total = train_targets_tensor.size(0)
                        train_accuracy = correct / total if total > 0 else 0.0
                        
                    # Calculate generalization as the difference between train and validation accuracy
                    accuracy_gap = abs(train_accuracy - ensemble_accuracy)
                    generalization_score = 1.0 - min(1.0, accuracy_gap / max(0.001, train_accuracy))
                else:
                    # Default if we couldn't get training samples
                    generalization_score = 0.5
                    logger.warning("Couldn't calculate generalization score from training data. Using default.")
            except Exception as gen_err:
                logger.error(f"Error calculating generalization score: {gen_err}")
                generalization_score = 0.5
                
            metrics["generalization_score"] = float(generalization_score)
            
            # 4. Calculate convergence rate if GA data is available
            if ga_convergence_data and "convergence_rate" in ga_convergence_data:
                metrics["convergence_rate"] = float(ga_convergence_data["convergence_rate"])
            elif hasattr(self, "ga_stacking") and hasattr(self.ga_stacking, "fitness_history"):
                # We have GA fitness history, calculate from there
                fitness_history = self.ga_stacking.fitness_history
                if len(fitness_history) > 1:
                    initial_fitness = fitness_history[0]
                    final_fitness = fitness_history[-1]
                    generations = len(fitness_history)
                    
                    # Calculate when we reach 90% of improvement
                    target_improvement = 0.9 * (final_fitness - initial_fitness)
                    gen_to_90_percent = generations  # Default
                    
                    for i, fitness in enumerate(fitness_history):
                        if (fitness - initial_fitness) >= target_improvement:
                            gen_to_90_percent = i + 1
                            break
                            
                    convergence_rate = 1.0 - (gen_to_90_percent / generations)
                    metrics["convergence_rate"] = float(max(0.0, min(1.0, convergence_rate)))
                else:
                    metrics["convergence_rate"] = 0.5  # Default
            else:
                metrics["convergence_rate"] = 0.5  # Default
            
            # 5. Calculate average performance of base models
            base_accuracies = []
            for model in base_models:
                try:
                    with torch.no_grad():
                        outputs = model(X_val)
                        outputs = outputs.to(dtype=torch.float32)
                        # Ensure dimensions match
                        if outputs.shape != y_val.shape:
                            if len(outputs.shape) == 1:
                                outputs = outputs.unsqueeze(1)
                                
                        correct = torch.sum(torch.abs(outputs - y_val) < threshold).item()
                        total = y_val.size(0)
                        accuracy = correct / total if total > 0 else 0.0
                        base_accuracies.append(accuracy)
                except Exception as e:
                    logger.warning(f"Error evaluating base model: {e}")
                    # Skip this model
            
            if base_accuracies:
                avg_base_model_score = sum(base_accuracies) / len(base_accuracies)
            else:
                avg_base_model_score = 0.0
                    
            metrics["avg_base_model_score"] = float(avg_base_model_score)
            
            # Final score calculation according to GA-Stacking reward formula
            weights = {
                "ensemble_accuracy": 0.40,
                "diversity_score": 0.20,
                "generalization_score": 0.20,
                "convergence_rate": 0.10,
                "avg_base_model_score": 0.10
            }
            
            weighted_score = (
                metrics["ensemble_accuracy"] * weights["ensemble_accuracy"] +
                metrics["diversity_score"] * weights["diversity_score"] +
                metrics["generalization_score"] * weights["generalization_score"] +
                metrics["convergence_rate"] * weights["convergence_rate"] +
                metrics["avg_base_model_score"] * weights["avg_base_model_score"]
            )
            
            # Apply bonus for exceptional accuracy (>90%)
            if metrics["ensemble_accuracy"] > 0.9:
                bonus = (metrics["ensemble_accuracy"] - 0.9) * 1.5
                weighted_score += bonus
                metrics["accuracy_bonus"] = float(bonus)
            
            # Convert to integer score (0-10000)
            metrics["final_score"] = int(min(1.0, weighted_score) * 10000)
            
            # Log the metrics
            logger.info(f"GA-Stacking metrics: Accuracy={metrics['ensemble_accuracy']:.4f}, "
                    f"Diversity={metrics['diversity_score']:.4f}, Final Score={metrics['final_score']}")
            
        except Exception as e:
            logger.error(f"Error calculating GA-Stacking metrics: {e}")
            # Provide default values
            metrics = {
                "ensemble_accuracy": 0.0,
                "diversity_score": 0.0,
                "generalization_score": 0.0,
                "convergence_rate": 0.5,
                "avg_base_model_score": 0.0,
                "final_score": 0
            }
        
        return metrics

    def _calculate_model_diversity(self, validation_data, base_models=None, base_model_names=None):
        """
        Calculate diversity among base models with error handling.
        
        Args:
            validation_data: Tuple of (X_val, y_val)
            base_models: Optional list of base models (if None, uses self.base_models)
            base_model_names: Optional list of model names
            
        Returns:
            float: Diversity score (0-1)
        """
        try:
            X_val, _ = validation_data
            
            if X_val.numel() == 0:
                logger.warning("Empty validation data. Cannot calculate diversity.")
                return 0.0
            
            # Use provided models or default to self.base_models
            models_to_use = base_models if base_models is not None else self.base_models
            model_names = base_model_names if base_model_names is not None else self.model_names
            
            # Ensure data is float32
            X_val = X_val.to(dtype=torch.float32, device=self.device)
                
            # Get predictions from all models
            all_predictions = []
            
            for i, model in enumerate(models_to_use):
                model_name = model_names[i] if i < len(model_names) else f"model_{i}"
                
                # Skip meta-learners in diversity calculation
                if model_name == "meta_lr" or "meta" in model_name.lower():
                    continue
                    
                try:
                    with torch.no_grad():
                        model.eval()
                        outputs = model(X_val)
                        outputs = outputs.to(dtype=torch.float32)
                        
                        # For regression, discretize predictions for diversity calculation
                        # We'll round to the nearest 0.5 for comparison
                        discretized = torch.round(outputs * 2) / 2
                        all_predictions.append(discretized)
                except Exception as e:
                    logger.warning(f"Error getting predictions for diversity calculation from {model_name}: {e}")
                    # Skip this model
            
            # Need at least 2 models with predictions
            if len(all_predictions) < 2:
                logger.warning("Not enough models with valid predictions for diversity calculation")
                return 0.0
            
            # Calculate pairwise disagreement
            disagreement_sum = 0
            comparison_count = 0
            
            for i in range(len(all_predictions)):
                for j in range(i+1, len(all_predictions)):
                    try:
                        # Ensure tensors have same shape
                        if all_predictions[i].shape != all_predictions[j].shape:
                            if len(all_predictions[i].shape) == 1:
                                all_predictions[i] = all_predictions[i].unsqueeze(1)
                            if len(all_predictions[j].shape) == 1:
                                all_predictions[j] = all_predictions[j].unsqueeze(1)
                                
                        # Count how often models i and j make different predictions
                        disagreement = torch.mean((all_predictions[i] != all_predictions[j]).float()).item()
                        disagreement_sum += disagreement
                        comparison_count += 1
                    except Exception as e:
                        logger.warning(f"Error in diversity calculation for models {i} and {j}: {e}")
                        # Skip this comparison
            
            # Normalize by number of comparisons
            if comparison_count == 0:
                return 0.0
            
            return float(disagreement_sum / comparison_count)
                
        except Exception as e:
            logger.error(f"Error in model diversity calculation: {e}")
            return 0.0
    
    def _calculate_ga_convergence_rate(self):
        """
        Calculate how quickly the GA algorithm converged to a solution.
        
        Returns:
            float: Convergence rate (0-1)
        """
        # Get fitness history from GA optimizer
        if not hasattr(self, "ga_optimizer") or not hasattr(self.ga_optimizer, "fitness_history"):
            return 0.5  # Default value
        
        fitness_history = self.ga_optimizer.fitness_history
        
        if not fitness_history or len(fitness_history) < 2:
            return 0.5  # Not enough data
        
        # Calculate the rate of improvement
        initial_fitness = fitness_history[0]
        final_fitness = fitness_history[-1]
        max_generations = len(fitness_history)
        
        # When did we reach 90% of the final improvement?
        target_improvement = 0.9 * (final_fitness - initial_fitness)
        generations_to_90_percent = max_generations  # Default
        
        for i, fitness in enumerate(fitness_history):
            if (fitness - initial_fitness) >= target_improvement:
                generations_to_90_percent = i + 1
                break
        
        # Normalize: faster convergence = higher score
        convergence_rate = 1.0 - (generations_to_90_percent / max_generations)
        
        # Ensure bounds
        return max(0.0, min(1.0, convergence_rate))

    def _create_validation_split(self, validation_split=0.2):
        """
        Create a validation split from the training data.
        
        Args:
            validation_split: Fraction of training data to use for validation
            
        Returns:
            tuple: (train_data, validation_data)
        """
        try:
            # Get the dataset from the data loader
            dataset = self.train_loader.dataset
            
            # Check what kind of dataset we're working with
            if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
                # Standard dataset with data and targets attributes
                train_data = dataset.data
                train_targets = dataset.targets
            elif isinstance(dataset, torch.utils.data.TensorDataset):
                # TensorDataset case
                train_data = dataset.tensors[0]
                train_targets = dataset.tensors[1]
            elif hasattr(dataset, 'x') and hasattr(dataset, 'y'):
                # Our SimpleDataset case
                train_data = dataset.x
                train_targets = dataset.y
            else:
                # Generic case: get all samples
                all_data = []
                all_targets = []
                for data, target in self.train_loader:
                    all_data.append(data)
                    all_targets.append(target)
                
                train_data = torch.cat(all_data, dim=0)
                train_targets = torch.cat(all_targets, dim=0)
                
            # Calculate split
            num_samples = len(train_data)
            num_val_samples = int(num_samples * validation_split)
            
            # Create random indices
            indices = torch.randperm(num_samples)
            val_indices = indices[:num_val_samples]
            train_indices = indices[num_val_samples:]
            
            # Extract validation data
            val_data = train_data[val_indices]
            val_targets = train_targets[val_indices]
            
            if len(val_data) > 500:
                # Chỉ lấy tối đa 500 mẫu cho validation để tăng tốc GA
                random_indices = torch.randperm(len(val_data))[:500]
                val_data = val_data[random_indices]
                val_targets = val_targets[random_indices]
                logger.info(f"Reduced validation set to 500 samples for faster GA training")
            
            # Extract training data (subset without validation)
            train_data_subset = train_data[train_indices]
            train_targets_subset = train_targets[train_indices]
            
            logger.info(f"Created validation split: {len(val_data)} validation, {len(train_data_subset)} training samples")
            
            return (train_data_subset, train_targets_subset), (val_data, val_targets)
        
        except Exception as e:
            logger.error(f"Error creating validation split: {e}")
            logger.info("Falling back to simple random split of DataLoader")
            
            # Fallback method using random_split
            train_dataset = self.train_loader.dataset
            
            # Calculate sizes
            val_size = int(len(train_dataset) * validation_split)
            train_size = len(train_dataset) - val_size
            
            # Split the dataset
            train_subset, val_subset = random_split(
                train_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
            
            # Create new data loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=self.train_loader.batch_size,
                shuffle=True
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=self.train_loader.batch_size,
                shuffle=False
            )
            
            # Convert to tensor format for consistency
            all_train_data = []
            all_train_targets = []
            all_val_data = []
            all_val_targets = []
            
            for data, target in train_loader:
                all_train_data.append(data)
                all_train_targets.append(target)
                
            for data, target in val_loader:
                all_val_data.append(data)
                all_val_targets.append(target)
            
            if all_train_data and all_val_data:
                train_data_tensor = torch.cat(all_train_data, dim=0)
                train_targets_tensor = torch.cat(all_train_targets, dim=0)
                val_data_tensor = torch.cat(all_val_data, dim=0)
                val_targets_tensor = torch.cat(all_val_targets, dim=0)
                
                logger.info(f"Created fallback validation split: {len(val_data_tensor)} validation, {len(train_data_tensor)} training samples")
                
                return (train_data_tensor, train_targets_tensor), (val_data_tensor, val_targets_tensor)
            else:
                # If all else fails, return empty tensors with warning
                logger.warning("Failed to create validation split, returning empty tensors")
                empty_tensor = torch.tensor([])
                return (empty_tensor, empty_tensor), (empty_tensor, empty_tensor)

    def _verify_and_fix_models(self):
        """Verify models have correct dimensions and fix if needed."""
        if not self.base_models:
            logger.warning("No base models to verify")
            return
            
        # First pass: count base models and find meta learner
        meta_model_idx = []
        base_models_count = 0
        
        for i, model_name in enumerate(self.model_names):
            if model_name == "meta_lr" or "MetaLearner" in model_name:
                meta_model_idx.append(i)
            else:
                base_models_count += 1
        
        logger.info(f"Identified {base_models_count} base models and {len(meta_model_idx)} meta-learners")
        
        # Second pass: fix each model
        for i, (model, name) in enumerate(zip(self.base_models, self.model_names)):
            # Fix meta learners
            if i in meta_model_idx:
                if hasattr(model, 'input_dim') and model.input_dim != base_models_count:
                    logger.warning(f"Fixing meta learner input dimension: {model.input_dim} -> {base_models_count}")
                    
                    # Check if it's a pytorch model with linear layer
                    if hasattr(model, 'linear') and hasattr(model.linear, 'weight'):
                        with torch.no_grad():
                            # Create new linear layer with correct dimensions
                            new_linear = nn.Linear(base_models_count, model.linear.weight.shape[0], 
                                                device=model.linear.weight.device)
                            # Initialize with uniform weights
                            new_linear.weight.data.fill_(1.0 / base_models_count)
                            # Replace the layer
                            model.linear = new_linear
                            model.input_dim = base_models_count
                            logger.info(f"Fixed meta learner linear layer dimensions")
                    elif hasattr(model, 'model') and hasattr(model.model, 'coef_'):
                        # For sklearn models
                        try:
                            # Initialize with equal weights
                            model.model.coef_ = np.ones((1, base_models_count)) / base_models_count
                            model.input_dim = base_models_count
                            logger.info(f"Fixed sklearn meta learner coefficients")
                        except Exception as e:
                            logger.error(f"Error fixing sklearn meta learner: {e}")
            
            # Fix input dimensions for base models if needed
            elif hasattr(model, 'input_dim') and model.input_dim != self.input_dim:
                logger.warning(f"Input dimension mismatch for {name}: {model.input_dim} != {self.input_dim}")
                
                # For PyTorch models with linear layer
                if hasattr(model, 'linear') and hasattr(model.linear, 'weight'):
                    try:
                        with torch.no_grad():
                            # Create new layer with correct dimensions
                            new_linear = nn.Linear(self.input_dim, model.linear.weight.shape[0], 
                                                device=model.linear.weight.device)
                            # Initialize with small random values
                            torch.nn.init.xavier_uniform_(new_linear.weight)
                            # Replace the layer
                            model.linear = new_linear
                            model.input_dim = self.input_dim
                            logger.info(f"Fixed {name} input dimension")
                    except Exception as e:
                        logger.error(f"Error fixing {name} input dimension: {e}")
    
    def check_and_fix_zero_coefficients(self):
        """Check models for zero coefficients and initialize them if needed."""
        for i, (model, name) in enumerate(zip(self.base_models, self.model_names)):
            if hasattr(model, 'model') and hasattr(model.model, 'coef_'):
                # For sklearn wrappers
                if hasattr(model.model, 'coef_') and np.all(np.array(model.model.coef_) == 0):
                    logger.warning(f"Model {name} has all-zero coefficients before training, initializing")
                    model.model.coef_ = np.random.normal(0, 0.01, model.model.coef_.shape)
                    if hasattr(model.model, 'intercept_'):
                        model.model.intercept_ = np.random.normal(0, 0.01, model.model.intercept_.shape)
                    model.is_initialized = False  # Force retraining
    
    def evaluate(
        self, parameters: Parameters, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on the local test dataset with proper meta-learner handling.
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
                # Continue anyway, server will check again
        
        # Get global model from IPFS if hash is provided
        ipfs_hash = config.get("ipfs_hash", None)
        round_num = config.get("server_round", 0)
        
        if ipfs_hash:
            self.set_parameters_from_ipfs(ipfs_hash)
        else:
            # Fallback to direct parameters
            params_arrays = parameters_to_ndarrays(parameters)
            self.set_parameters_individual(params_arrays)
        
        # IMPORTANT: Verify model dimensions match after loading
        self._verify_and_fix_models()
        
        # Evaluate the ensemble
        accuracy, loss, metrics = self._evaluate_ensemble(self.test_loader)
        
        # Add to metrics history
        self.metrics_history.append({
            "round": round_num,
            "eval_loss": float(loss),
            "accuracy": float(accuracy),
            # Add fraud-specific metrics
            "precision": float(metrics['precision']),
            "recall": float(metrics['recall']),
            "f1_score": float(metrics['f1_score']),
            "specificity": float(metrics.get('specificity', 0.0)),
            "auc_roc": float(metrics.get('auc_roc', 0.5)),
            # Include confusion matrix elements
            "true_positives": int(metrics.get('true_positives', 0)),
            "false_positives": int(metrics.get('false_positives', 0)),
            "true_negatives": int(metrics.get('true_negatives', 0)),
            "false_negatives": int(metrics.get('false_negatives', 0)),
            "eval_samples": len(self.test_loader.dataset),
            "ipfs_hash": ipfs_hash,
            "wallet_address": self.wallet_address if self.wallet_address else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "wallet_address": self.wallet_address if self.wallet_address else "unknown",
            "ensemble_size": len(self.base_models) if self.ensemble_model else 0
        }
        
        return float(loss), len(self.test_loader.dataset), metrics
    
    def _evaluate_ensemble(self, data_loader: DataLoader) -> Tuple[float, float, Dict]:
        """
        Đánh giá mô hình ensemble với cải thiện cho bài toán phát hiện gian lận.
        Bao gồm tìm ngưỡng tối ưu tự động và các chỉ số đánh giá phù hợp với dữ liệu mất cân bằng.
        """
        if self.ensemble_model is None:
            logger.warning("No ensemble model for evaluation, using equal weights")
            # Create ensemble with equal weights if not available
            equal_weights = np.ones(len(self.base_models)) / len(self.base_models)
            self.ensemble_model = EnsembleModel(
                models=self.base_models,
                weights=equal_weights,
                model_names=self.model_names,
                device=self.device
            )
        
        self.ensemble_model.eval()
        
        # Sử dụng các list để thu thập dự đoán và nhãn thực
        all_targets = []
        all_outputs = []  # Raw outputs (probabilities)
        
        # Sử dụng weighted BCELoss cho dữ liệu mất cân bằng
        # pos_weight càng cao thì phạt các false negatives nặng hơn
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.0], device=self.device))
        
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Sử dụng ensemble model
                output = self.ensemble_model(data)
                
                # Đảm bảo output và target có cùng shape
                if output.shape != target.shape:
                    if len(output.shape) == 1:
                        output = output.unsqueeze(1)
                    if len(target.shape) == 1:
                        target = target.unsqueeze(1)
                        
                # Tính loss
                loss = criterion(output, target)
                test_loss += loss.item() * len(data)
                
                # Lưu kết quả
                all_targets.append(target.cpu())
                all_outputs.append(output.cpu())
                
                total += target.size(0)
        
        # Tính loss trung bình
        avg_loss = test_loss / total if total > 0 else float('inf')
        
        # Gộp kết quả từ các batch
        y_true = torch.cat(all_targets).detach().numpy()
        y_score = torch.cat(all_outputs).detach().numpy()
        
        # Điều chỉnh shape
        if len(y_true.shape) > 1 and y_true.shape[1] == 1:
            y_true = y_true.flatten()
        if len(y_score.shape) > 1 and y_score.shape[1] == 1:
            y_score = y_score.flatten()
        
        # Tính toán các metrics
        try:
            # Log số lượng mẫu dương tính trong ground truth
            num_positive_true = np.sum(y_true == 1)
            logger.info(f"Positive samples in ground truth: {num_positive_true}/{len(y_true)} ({num_positive_true/len(y_true)*100:.2f}%)")
            
            # Khởi tạo dictionary chứa kết quả
            metrics = {}
            
            # ===== CẢI TIẾN 1: Tìm ngưỡng tối ưu tự động =====
            # Tìm ngưỡng tối ưu bằng cách tối đa hóa F1-score
            if num_positive_true > 0:
                # Thử nhiều ngưỡng khác nhau để tìm ngưỡng tốt nhất
                thresholds = np.linspace(0.001, 0.1, 100)  # Thử 50 ngưỡng từ 0.001 đến 0.5
                best_f1 = 0
                best_threshold = 0.01  # Mặc định là 0.01
                best_precision = 0
                best_recall = 0
                best_tp = 0
                best_fp = 0
                best_tn = 0
                best_fn = 0
                
                # Tạo dict để lưu trữ kết quả ở các ngưỡng khác nhau cho vẽ đồ thị
                threshold_results = {
                    'thresholds': thresholds,
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'positive_rate': []
                }
                
                # Tìm ngưỡng tối ưu
                for t in thresholds:
                    # Apply threshold
                    y_pred_t = (y_score >= t).astype(float)
                    
                    # Tính confusion matrix
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t, labels=[0, 1]).ravel()
                    
                    # Tính precision và recall
                    precision_t = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall_t = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity_t = tn / (tn + fp) if (tn + fp) > 0 else 0
                    f1_t = 2 * precision_t * recall_t / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0
                    
                    # Lưu kết quả cho các ngưỡng khác nhau
                    threshold_results['precision'].append(precision_t)
                    threshold_results['recall'].append(recall_t)
                    threshold_results['f1'].append(f1_t)
                    threshold_results['positive_rate'].append((tp + fp) / (tp + fp + tn + fn))
                    
                    # Cập nhật ngưỡng tốt nhất dựa vào F1
                    if f1_t > best_f1:
                        best_f1 = f1_t
                        best_threshold = t
                        best_precision = precision_t
                        best_recall = recall_t
                        best_tp = tp
                        best_fp = fp
                        best_tn = tn
                        best_fn = fn
                        
                # Lưu thông tin ngưỡng tốt nhất
                metrics['optimal_threshold'] = best_threshold
                metrics['threshold_results'] = threshold_results
                        
                logger.info(f"Optimal threshold found: {best_threshold:.6f} with F1: {best_f1:.4f}")
                logger.info(f"At optimal threshold - Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")
                logger.info(f"Confusion matrix at optimal threshold - TP: {best_tp}, FP: {best_fp}, TN: {best_tn}, FN: {best_fn}")
                
                # Sử dụng ngưỡng tối ưu cho dự đoán cuối cùng
                y_pred = (y_score >= best_threshold).astype(float)
                positive_preds = np.sum(y_pred == 1)
                logger.info(f"Positive predictions made: {positive_preds}/{len(y_pred)} ({positive_preds/len(y_pred)*100:.2f}%)")
                
                # Tính confusion matrix với ngưỡng tối ưu
                tn, fp, fn, tp = best_tn, best_fp, best_fn, best_tp
            else:
                # Không có mẫu dương tính trong ground truth
                logger.warning("No positive samples in ground truth, using default threshold")
                y_pred = (y_score >= 0.01).astype(float)  # Default threshold
                
                # Kiểm tra xem có dự đoán dương tính nào không
                positive_preds = np.sum(y_pred == 1)
                logger.info(f"Positive predictions made: {positive_preds}/{len(y_pred)} ({positive_preds/len(y_pred)*100:.2f}%)")
                
                if positive_preds > 0:
                    # Nếu có dự đoán dương tính nhưng không có mẫu dương tính thực, tất cả là false positives
                    tn = len(y_true) - positive_preds
                    fp = positive_preds
                    fn = 0
                    tp = 0
                else:
                    # Không có dự đoán dương tính và không có mẫu dương tính thực: tất cả true negatives
                    tn = len(y_true)
                    fp = 0
                    fn = 0
                    tp = 0
                    
                metrics['optimal_threshold'] = 0.01  # Default
            
            # ===== CẢI TIẾN 2: Tính toán các chỉ số đánh giá phù hợp cho dữ liệu mất cân bằng =====
            # Tính các chỉ số cơ bản
            accuracy = 100.0 * (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Tính chỉ số F2 (nhấn mạnh recall hơn precision)
            beta = 2
            f_beta = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0.0
            
            # ===== CẢI TIẾN 3: Thêm AUC-ROC và AUC-PR =====
            # AUC-ROC
            auc_roc = 0.5  # Mặc định
            if len(np.unique(y_true)) > 1:
                try:
                    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
                    auc_roc = roc_auc_score(y_true, y_score)
                    
                    # AUC-PR (Area Under Precision-Recall Curve) - tốt hơn AUC-ROC cho dữ liệu mất cân bằng
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
                    auc_pr = auc(recall_curve, precision_curve)
                    metrics["auc_pr"] = auc_pr
                    
                    logger.info(f"AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")
                except Exception as e:
                    logger.warning(f"Error calculating AUC metrics: {e}")
            
            # ===== CẢI TIẾN 4: Thêm các chỉ số đánh giá mở rộng =====
            # Matthews Correlation Coefficient - chỉ số tốt cho dữ liệu mất cân bằng
            mcc_numerator = (tp * tn) - (fp * fn)
            mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 1
            mcc = mcc_numerator / mcc_denominator
            
            # Tính Balanced Accuracy - trung bình của Sensitivity và Specificity
            balanced_acc = (recall + specificity) / 2 if (recall + specificity) > 0 else 0.0
            
            # Thêm vào metrics dict
            metrics.update({
                "accuracy": accuracy / 100.0,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1_score": f1,
                "f2_score": f_beta,  # Thêm F2-score
                "auc_roc": auc_roc,
                "mcc": mcc,  # Thêm Matthews Correlation Coefficient
                "balanced_accuracy": balanced_acc,  # Thêm Balanced Accuracy
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "positive_rate_groundtruth": num_positive_true / len(y_true) if len(y_true) > 0 else 0,
                "positive_rate_predicted": np.sum(y_pred == 1) / len(y_pred) if len(y_pred) > 0 else 0
            })
            
            # Log kết quả đánh giá chính
            logger.info(f"Ensemble evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                    f"F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")
                    
        except Exception as e:
            # Xử lý ngoại lệ
            logger.error(f"Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to basic metrics
            metrics = {
                "accuracy": (np.sum(y_score >= 0.5) == y_true).mean() if len(y_true) > 0 else 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc_roc": 0.5,
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0
            }
        
        return accuracy, avg_loss, metrics
    
    def _evaluate_single_model(self, model: nn.Module, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate a single model.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader to evaluate on
            
        Returns:
            Tuple of (accuracy, loss)
        """
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item() * len(data)
                
                # For regression, we consider a prediction correct if it's within a threshold
                threshold = 0.1
                correct += torch.sum((torch.abs(output - target) < threshold)).item()
                total += target.size(0)
        
        # Calculate average loss and accuracy
        avg_loss = test_loss / total if total > 0 else float('inf')
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return accuracy, avg_loss
    
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
    input_dim: int = 10,
    output_dim: int = 1,
    ensemble_size: int = 8,
    device: str = "cpu",
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
        input_dim: Input dimension for the model
        output_dim: Output dimension for the model
        ensemble_size: Number of models in the ensemble
        device: Device to use for training ('cpu' or 'cuda')
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

    # Load dataset from files
    def load_dataset_from_files(train_file, test_file):
        """
        Load dataset from train and test files with preprocessing for categorical variables
        
        Args:
            train_file: Path to training data file
            test_file: Path to test data file
            
        Returns:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            input_dim: Input dimension for the model
            output_dim: Output dimension for the model
        """
        # Import required libraries
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        import numpy as np
        
        # Check if files exist
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
            
        # Read files with comment handling
        train_df = pd.read_csv(train_file, comment='#')
        test_df = pd.read_csv(test_file, comment='#')
        
        # Check for Credit Card Fraud dataset (Kaggle format)
        is_creditcard_dataset = False
        if 'V1' in train_df.columns and 'V2' in train_df.columns and 'V3' in train_df.columns:
            is_creditcard_dataset = True
            logger.info("Detected Credit Card Fraud dataset (Kaggle format)")
            
            # If the target column is "Class", rename it to "is_fraud" for consistency
            if 'Class' in train_df.columns:
                train_df = train_df.rename(columns={'Class': 'is_fraud'})
                test_df = test_df.rename(columns={'Class': 'is_fraud'})
        
        # Identify target column (fraud detection dataset uses 'is_fraud')
        if 'is_fraud' in train_df.columns:
            target_column = 'is_fraud'
        elif 'target' in train_df.columns:
            target_column = 'target'
        else:
            # Try to identify the target column
            potential_targets = ['label', 'class', 'fraud', 'Class']
            target_found = False
            
            for potential in potential_targets:
                if potential in train_df.columns:
                    target_column = potential
                    target_found = True
                    break
                    
            if not target_found:
                # Last resort: use the last column as target
                target_column = train_df.columns[-1]
                logger.warning(f"Could not identify target column, using the last column: {target_column}")
        
        # Extract features and target
        feature_columns = [col for col in train_df.columns if col != target_column]
        
        # For Credit Card dataset, special handling
        if is_creditcard_dataset:
            # PCA components (V1-V28) are already normalized, only need to normalize Amount
            if 'Amount' in train_df.columns:
                # Replace negative values with small positive value before log transform
                train_df['Amount'] = train_df['Amount'].clip(lower=0.0001)
                test_df['Amount'] = test_df['Amount'].clip(lower=0.0001)
                
                # Log-transform and standardize Amount
                train_df['Amount'] = np.log1p(train_df['Amount'])
                test_df['Amount'] = np.log1p(test_df['Amount'])
                
            # Drop Time column if it exists (we've already extracted temporal features)
            if 'Time' in train_df.columns:
                train_df = train_df.drop('Time', axis=1)
                test_df = test_df.drop('Time', axis=1)
                
            # Drop any non-feature columns
            for col in ['original_time', 'original_amount', 'cluster', 'amount_bin']:
                if col in train_df.columns:
                    train_df = train_df.drop(col, axis=1)
                if col in test_df.columns:
                    test_df = test_df.drop(col, axis=1)
                
            # Update feature columns after dropping
            feature_columns = [col for col in train_df.columns if col != target_column]
            
            # No categorical columns to encode in this dataset
            categorical_cols = []
            numerical_cols = feature_columns
        else:
            # For other datasets, identify categorical and numerical columns
            categorical_cols = []
            numerical_cols = []
            
            for col in feature_columns:
                if train_df[col].dtype == 'object' or train_df[col].dtype.name == 'category':
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
        
        logger.info(f"Dataset features: {len(feature_columns)} total, "
                f"{len(categorical_cols)} categorical, {len(numerical_cols)} numerical")
        
        # Create preprocessing pipeline if needed
        if not is_creditcard_dataset:
            if categorical_cols:
                # We have categorical columns to handle
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numerical_cols),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                    ])
                
                # Fit preprocessor on training data
                preprocessor.fit(train_df[feature_columns])
                
                # Transform data
                X_train = preprocessor.transform(train_df[feature_columns])
                X_test = preprocessor.transform(test_df[feature_columns])
            else:
                # Only numerical data, just standardize
                scaler = StandardScaler()
                X_train = scaler.fit_transform(train_df[feature_columns])
                X_test = scaler.transform(test_df[feature_columns])
        else:
            # For Credit Card dataset, features are already processed
            X_train = train_df[feature_columns].values
            X_test = test_df[feature_columns].values
            
        # Convert sparse matrix to dense if needed
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
        
        # Get targets
        y_train = train_df[target_column].values
        y_test = test_df[target_column].values
        
        # Reshape targets to be 2D if needed
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
        
        # Convert to PyTorch tensors
        train_x = torch.tensor(X_train, dtype=torch.float32)
        train_y = torch.tensor(y_train, dtype=torch.float32)
        test_x = torch.tensor(X_test, dtype=torch.float32)
        test_y = torch.tensor(y_test, dtype=torch.float32)
        
        # Get dimensions
        input_dim = train_x.shape[1]
        output_dim = train_y.shape[1]
        
        logger.info(f"Dataset loaded - Input dim: {input_dim}, Output dim: {output_dim}")
        logger.info(f"Train samples: {len(train_x)}, Test samples: {len(test_x)}")
        
        # Check for class imbalance if it's a classification task
        if output_dim == 1:
            pos_ratio = train_y.mean().item() * 100
            logger.info(f"Positive class ratio: {pos_ratio:.2f}%")
        
        # Create data loaders
        class FraudDataset(Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y
            
            def __len__(self):
                return len(self.x)
            
            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]
        
        train_loader = DataLoader(
            FraudDataset(train_x, train_y), batch_size=32, shuffle=True
        )
        test_loader = DataLoader(
            FraudDataset(test_x, test_y), batch_size=32, shuffle=False
        )
        
        return train_loader, test_loader, input_dim, output_dim
    # Load dataset
    try:
        train_loader, test_loader, data_input_dim, data_output_dim = load_dataset_from_files(train_file, test_file)
        
        # Override input_dim and output_dim with actual values from data if not explicitly set
        if input_dim == 10:  # This is the default value
            input_dim = data_input_dim
            logger.info(f"Using input dimension from dataset: {input_dim}")
        
        if output_dim == 1:  # This is the default value for fraud detection
            output_dim = data_output_dim
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # Create client
    client = GAStackingClient(
        input_dim=input_dim,
        output_dim=output_dim,
        train_loader=train_loader,
        test_loader=test_loader,
        ensemble_size=ensemble_size,
        ipfs_connector=ipfs_connector,
        blockchain_connector=blockchain_connector,
        wallet_address=wallet_address,
        private_key=private_key,
        device=device,
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
    parser.add_argument("--input-dim", type=int, default=31, help="Input dimension for the model")
    parser.add_argument("--output-dim", type=int, default=1, help="Output dimension for the model")
    parser.add_argument("--ensemble-size", type=int, default=8, help="Number of models in the ensemble")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for training")
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
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        ensemble_size=args.ensemble_size,
        device=args.device,
        ga_generations=args.ga_generations,
        ga_population_size=args.ga_population_size
    )