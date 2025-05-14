"""
Ensemble aggregation strategy for federated learning with GA-Stacking.
Handles aggregation of ensemble models from multiple clients.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
from ga_stacking_reward_system import GAStackingRewardSystem


import torch
import torch.nn as nn

from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

logger = logging.getLogger("Ensemble-Aggregation")


class EnsembleAggregator:
    """Aggregates ensemble models from multiple clients in federated learning."""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the ensemble aggregator.
        
        Args:
            device: Device to use for computations
        """
        self.device = torch.device(device)
    
    def deserialize_ensemble(self, parameters: List[np.ndarray]) -> Dict[str, Any]:
        """
        Deserialize ensemble model from parameters.
        
        Args:
            parameters: List of parameter arrays
            
        Returns:
            Deserialized ensemble state
        """
        if len(parameters) != 1 or parameters[0].dtype != np.uint8:
            logger.warning("Parameters don't appear to be a serialized ensemble")
            return None
        
        try:
            # Convert bytes to ensemble state
            ensemble_bytes = parameters[0].tobytes()
            ensemble_state = json.loads(ensemble_bytes.decode('utf-8'))
            return ensemble_state
        except Exception as e:
            logger.error(f"Failed to deserialize ensemble state: {e}")
            return None
    
    def serialize_ensemble(self, ensemble_state: Dict[str, Any]) -> List[np.ndarray]:
        """
        Serialize ensemble state to parameters.
        
        Args:
            ensemble_state: Ensemble state dictionary
            
        Returns:
            List of parameter arrays
        """
        try:
            # Convert ensemble state to bytes
            ensemble_bytes = json.dumps(ensemble_state).encode('utf-8')
            return [np.frombuffer(ensemble_bytes, dtype=np.uint8)]
        except Exception as e:
            logger.error(f"Failed to serialize ensemble state: {e}")
            return None
    
    def aggregate_ensembles(
        self, 
        ensembles: List[Dict[str, Any]], 
        weights: List[float]
    ) -> Dict[str, Any]:
        """
        Aggregate multiple ensemble models.
        
        Args:
            ensembles: List of ensemble state dictionaries
            weights: Weights for each ensemble
            
        Returns:
            Aggregated ensemble state
        """
        if not ensembles:
            logger.error("No ensembles to aggregate")
            return None
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            logger.warning("Total weight is zero, using equal weights")
            weights = [1.0/len(ensembles)] * len(ensembles)
        else:
            weights = [w/total_weight for w in weights]
        
        # Get all unique model names across all ensembles
        all_model_names = set()
        for ensemble in ensembles:
            all_model_names.update(ensemble["model_names"])
        
        # Create mapping from model name to index for each ensemble
        name_to_idx = []
        for ensemble in ensembles:
            mapping = {name: idx for idx, name in enumerate(ensemble["model_names"])}
            name_to_idx.append(mapping)
        
        # Collect all model state dicts by model name
        model_states_by_name = defaultdict(list)
        model_weights_by_name = defaultdict(list)
        
        for i, ensemble in enumerate(ensembles):
            ensemble_weight = weights[i]
            for name, idx in name_to_idx[i].items():
                model_states_by_name[name].append(ensemble["model_state_dicts"][idx])
                # Calculate effective weight = ensemble_weight * model_weight
                model_weight = ensemble["weights"][idx] * ensemble_weight
                model_weights_by_name[name].append(model_weight)
        
        for ensemble in ensembles:
            # Create a consistent order mapping
            model_order = {name: i for i, name in enumerate(ensemble["model_names"])}
            
            # Sort everything by model name to ensure consistent ordering
            sorted_indices = sorted(range(len(ensemble["model_names"])), key=lambda i: ensemble["model_names"][i])
            ensemble["model_names"] = [ensemble["model_names"][i] for i in sorted_indices]
            ensemble["weights"] = [ensemble["weights"][i] for i in sorted_indices]
            ensemble["model_state_dicts"] = [ensemble["model_state_dicts"][i] for i in sorted_indices]
        
        # Aggregate model state dicts
        aggregated_models = []
        aggregated_names = []
        aggregated_weights = []
        
        for name in sorted(model_states_by_name.keys()):
            # Get all states for this model
            states = model_states_by_name[name]
            model_weights = model_weights_by_name[name]
            
            # Normalize weights for this model
            total_weight = sum(model_weights)
            if total_weight > 0:
                normalized_weights = [w/total_weight for w in model_weights]
            else:
                normalized_weights = [1.0/len(states)] * len(states)
            
            # Aggregate state dicts
            aggregated_state = self._aggregate_state_dicts(states, normalized_weights)
            
            if aggregated_state:
                aggregated_models.append(aggregated_state)
                aggregated_names.append(name)
                # Use average of model weights across ensembles as starting point
                avg_weight = sum(model_weights) / len(model_weights) if model_weights else 0.0
                aggregated_weights.append(avg_weight)
        
        # Normalize final weights
        total_weight = sum(aggregated_weights)
        if total_weight > 0:
            aggregated_weights = [w/total_weight for w in aggregated_weights]
        else:
            aggregated_weights = [1.0/len(aggregated_models)] * len(aggregated_models)
        
        # Create aggregated ensemble state
        aggregated_ensemble = {
            "weights": aggregated_weights,
            "model_names": aggregated_names,
            "model_state_dicts": aggregated_models
        }
        
        return aggregated_ensemble
    
    def _aggregate_state_dicts(
        self, 
        state_dicts: List[Dict[str, Any]], 
        weights: List[float]
    ) -> Dict[str, Any]:
        """
        Aggregate multiple state dictionaries.
        
        Args:
            state_dicts: List of state dictionaries
            weights: Weight for each state dict
            
        Returns:
            Aggregated state dictionary
        """
        # Check if we have anything to aggregate
        if not state_dicts:
            return None
        
        # If only one state dict, just return it
        if len(state_dicts) == 1:
            return state_dicts[0]
        
        # Get all keys from all state dicts
        all_keys = set()
        for sd in state_dicts:
            all_keys.update(sd.keys())
        
        # Initialize aggregated state dict
        aggregated = {}
        
        # Aggregate each parameter
        for key in all_keys:
            # Check which state dicts have this key
            valid_dicts = []
            valid_weights = []
            
            for i, sd in enumerate(state_dicts):
                if key in sd:
                    valid_dicts.append(sd[key])
                    valid_weights.append(weights[i])
            
            # Skip if no valid dicts
            if not valid_dicts:
                continue
            
            # Normalize weights
            total_weight = sum(valid_weights)
            if total_weight > 0:
                normalized_weights = [w/total_weight for w in valid_weights]
            else:
                normalized_weights = [1.0/len(valid_dicts)] * len(valid_dicts)
            
            # Handle string parameters (like "estimator")
            if all(isinstance(param, str) for param in valid_dicts):
                # Take the most common string value
                from collections import Counter
                param_counter = Counter(valid_dicts)
                aggregated[key] = param_counter.most_common(1)[0][0]
                continue
            
            # Handle scalar parameters (int, float)
            if all(isinstance(param, (int, float)) for param in valid_dicts):
                # For integers, take the weighted average and round
                if all(isinstance(param, int) for param in valid_dicts):
                    weighted_avg = sum(param * w for param, w in zip(valid_dicts, normalized_weights))
                    aggregated[key] = int(round(weighted_avg))
                # For floats, take the weighted average
                else:
                    aggregated[key] = sum(param * w for param, w in zip(valid_dicts, normalized_weights))
                continue
            
            # Handle list parameters (including nested lists)
            try:
                # Try to convert all parameters to float arrays
                param_arrays = []
                for param in valid_dicts:
                    # Handle potential nested lists (e.g., for coef matrices)
                    if isinstance(param, list) and all(isinstance(item, list) for item in param):
                        # For nested lists, convert each inner list to float
                        param_array = np.array([
                            [float(x) if isinstance(x, (int, float, str)) else x for x in inner]
                            for inner in param
                        ], dtype=np.float64)
                    elif isinstance(param, list):
                        # For flat lists, convert directly to float
                        param_array = np.array([
                            float(x) if isinstance(x, (int, float, str)) else x for x in param
                        ], dtype=np.float64)
                    else:
                        # If it's not a list, create a scalar array
                        param_array = np.array([float(param)], dtype=np.float64)
                    
                    param_arrays.append(param_array)
                
                # Check shapes after conversion
                shapes = [arr.shape for arr in param_arrays]
                if len(set(shapes)) != 1:
                    logger.warning(f"Skipping parameter {key} due to shape mismatch: {shapes}")
                    # Use the parameter from the client with the highest weight
                    max_weight_idx = normalized_weights.index(max(normalized_weights))
                    aggregated[key] = valid_dicts[max_weight_idx]
                    continue
                
                # Aggregate parameter
                shape = shapes[0]
                aggregated_param = np.zeros(shape, dtype=np.float64)
                
                for i, param_array in enumerate(param_arrays):
                    aggregated_param += param_array * normalized_weights[i]
                
                # Convert back to list or scalar
                if len(shape) > 0:
                    aggregated[key] = aggregated_param.tolist()
                else:
                    # It's a scalar array, return as float
                    aggregated[key] = float(aggregated_param)
                
            except (ValueError, TypeError) as e:
                # If we can't convert to float arrays, this might be a non-numeric parameter
                logger.warning(f"Cannot aggregate parameter {key} numerically: {e}")
                # Use the parameter from the client with the highest weight
                max_weight_idx = normalized_weights.index(max(normalized_weights))
                aggregated[key] = valid_dicts[max_weight_idx]
        
            if key == 'coef':
                logger.info(f"Aggregating coefficients with shapes: {[np.array(d).shape if isinstance(d, list) else 'scalar' for d in valid_dicts]}")
                
                try:
                    # Convert all coef to numpy arrays with explicit shape checking
                    coef_arrays = []
                    coef_shapes = []
                    
                    for i, coef in enumerate(valid_dicts):
                        if isinstance(coef, list):
                            # Handle nested structure
                            coef_array = np.array(coef, dtype=np.float64)
                            coef_arrays.append(coef_array)
                            coef_shapes.append(coef_array.shape)
                            logger.debug(f"Client {i} coef shape: {coef_array.shape}, has non-zeros: {np.any(coef_array != 0)}")
                        else:
                            logger.warning(f"Client {i} has unexpected coef type: {type(coef)}")
                    
                    # Check shapes match
                    if len(set(str(s) for s in coef_shapes)) != 1:
                        logger.warning(f"Coefficient shape mismatch: {coef_shapes}, using weighted averaging per element")
                        
                        # Find max shape to accommodate all
                        max_dims = tuple(max(s[i] if i < len(s) else 0 for s in coef_shapes) for i in range(max(len(s) for s in coef_shapes)))
                        
                        # Reshape all arrays to max shape, padding with zeros
                        padded_arrays = []
                        for arr, w in zip(coef_arrays, normalized_weights):
                            padded = np.zeros(max_dims, dtype=np.float64)
                            slices = tuple(slice(0, min(dim, arr.shape[i])) for i, dim in enumerate(arr.shape))
                            padded[slices] = arr
                            padded_arrays.append(padded * w)
                        
                        # Sum all padded arrays
                        aggregated[key] = sum(padded_arrays).tolist()
                    else:
                        # Shapes match, do normal weighted average
                        aggregated[key] = (sum(arr * w for arr, w in zip(coef_arrays, normalized_weights))).tolist()
                        
                    # Check if result contains any non-zero values
                    if not np.any(np.array(aggregated[key]) != 0):
                        logger.warning(f"Aggregated coefficients are all zeros! Check client models.")
                        
                        # Fallback: use coefficients from client with highest weight
                        max_weight_idx = normalized_weights.index(max(normalized_weights))
                        aggregated[key] = valid_dicts[max_weight_idx]
                        logger.info(f"Falling back to coefficients from client with weight {normalized_weights[max_weight_idx]}")
                    
                except Exception as e:
                    logger.error(f"Error aggregating coefficients: {e}")
                    # Use coefficients from highest-weighted client
                    max_weight_idx = normalized_weights.index(max(normalized_weights))
                    aggregated[key] = valid_dicts[max_weight_idx]
                    logger.info(f"Exception - falling back to coefficients from client with weight {normalized_weights[max_weight_idx]}")
        
        return aggregated
    
    def aggregate_fit_results(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        weights: List[float]
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        """
        Aggregate fit results from multiple clients.
        
        Args:
            results: List of (client, fit_res) tuples
            weights: Weight for each result based on sample count
            
        Returns:
            Tuple of (parameters, metrics)
        """
        # Extract parameters and ensembles
        ensembles = []
        client_weights = []
        
        for i, (client, fit_res) in enumerate(results):
            try:
                # Convert parameters to ndarrays
                params = parameters_to_ndarrays(fit_res.parameters)
                
                # Deserialize ensemble
                ensemble = self.deserialize_ensemble(params)
                
                if ensemble:
                    ensembles.append(ensemble)
                    client_weights.append(weights[i])
                else:
                    logger.warning(f"Client {client.cid} did not return a valid ensemble")
            except Exception as e:
                logger.error(f"Error processing result from client {client.cid}: {e}")
        
        # Check if we have any valid ensembles
        if not ensembles:
            logger.error("No valid ensembles received from clients")
            return None, {"error": "no_valid_ensembles"}
        
        # Aggregate ensembles
        logger.info(f"Aggregating {len(ensembles)} ensembles")
        aggregated_ensemble = self.aggregate_ensembles(ensembles, client_weights)
        
        # Serialize aggregated ensemble
        parameters = self.serialize_ensemble(aggregated_ensemble)
        
        # Return as Flower parameters
        metrics = {
            "num_ensembles": len(ensembles),
            "num_models": len(aggregated_ensemble["model_names"]),
            "model_names": ",".join(aggregated_ensemble["model_names"]),
            "ensemble_weights": aggregated_ensemble["weights"]
        }
        
        return ndarrays_to_parameters(parameters), metrics
    
    def aggregate_evaluate_results(
        self,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        weights: List[float]
    ) -> Tuple[float, Dict[str, Scalar]]:
        """
        Aggregate evaluation results from multiple clients.
        
        Args:
            results: List of (client, evaluate_res) tuples
            weights: Weight for each result based on sample count
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Calculate weighted average of losses
        weighted_losses = 0.0
        weighted_metrics = {}
        total_weight = sum(weights)
        
        # Normalize weights
        if total_weight == 0:
            weights = [1.0/len(results)] * len(results)
            total_weight = 1.0
        else:
            weights = [w/total_weight for w in weights]
        
        # Collect all metrics
        all_metrics = defaultdict(list)
        all_weights = defaultdict(list)
        
        for i, (client, eval_res) in enumerate(results):
            weighted_losses += weights[i] * eval_res.loss
            
            # Collect metrics for aggregation
            for key, value in eval_res.metrics.items():
                try:
                    # Skip non-numeric metrics
                    if isinstance(value, (int, float)):
                        all_metrics[key].append(value)
                        all_weights[key].append(weights[i])
                except Exception as e:
                    logger.warning(f"Error processing metric {key}: {e}")
        
        # Aggregate metrics
        for key in all_metrics:
            values = all_metrics[key]
            metric_weights = all_weights[key]
            
            # Normalize weights
            total = sum(metric_weights)
            if total > 0:
                metric_weights = [w/total for w in metric_weights]
            else:
                metric_weights = [1.0/len(values)] * len(values)
            
            # Calculate weighted average
            weighted_metrics[key] = sum(v * w for v, w in zip(values, metric_weights))
        
        # Add number of clients that participated
        weighted_metrics["num_clients"] = len(results)
        
        return weighted_losses, weighted_metrics