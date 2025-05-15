"""
Ensemble aggregation strategy for federated learning with GA-Stacking.
Handles aggregation of ensemble models from multiple clients.
Uses scikit-learn models exclusively (no PyTorch).
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict

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
    
    def __init__(self):
        """
        Initialize the ensemble aggregator.
        """
        pass
    
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
        
        # Get all unique model names from all ensembles
        all_model_names = set()
        for ensemble in ensembles:
            all_model_names.update(ensemble.get("model_names", []))
        
        all_model_names = sorted(all_model_names)
        
        # Aggregate weights for each model
        aggregated_weights = np.zeros(len(all_model_names))
        weight_counts = np.zeros(len(all_model_names))
        
        for i, ensemble in enumerate(ensembles):
            ensemble_weight = weights[i]
            ensemble_model_names = ensemble.get("model_names", [])
            ensemble_weights = ensemble.get("weights", [])
            
            # Skip if ensemble is missing required data
            if not ensemble_model_names or not ensemble_weights:
                continue
                
            for j, model_name in enumerate(ensemble_model_names):
                if j < len(ensemble_weights):
                    # Find index in aggregated model names
                    agg_idx = all_model_names.index(model_name)
                    # Add weighted contribution
                    aggregated_weights[agg_idx] += ensemble_weights[j] * ensemble_weight
                    weight_counts[agg_idx] += 1
        
        # For models that weren't in all ensembles, adjust weights
        for i in range(len(all_model_names)):
            if weight_counts[i] > 0:
                # Normalize by the number of ensembles that had this model
                aggregated_weights[i] /= weight_counts[i]
        
        # Normalize weights
        if np.sum(aggregated_weights) > 0:
            aggregated_weights = aggregated_weights / np.sum(aggregated_weights)
        else:
            # Equal weights fallback
            aggregated_weights = np.ones(len(all_model_names)) / len(all_model_names)
        
        # We don't directly serialize the models, as we're using GA-Stacking
        # which trains models locally on each client. The global model just 
        # contains the ensemble structure and weights.
        
        # Create aggregated ensemble state
        aggregated_ensemble = {
            "model_names": list(all_model_names),
            "weights": aggregated_weights.tolist(),
            "ga_metadata": {
                "num_ensembles_aggregated": len(ensembles),
                "client_weights": weights
            }
        }
        
        return aggregated_ensemble
    
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
        client_ipfs_hashes = []
        client_metrics = []
        
        for i, (client, fit_res) in enumerate(results):
            try:
                # Convert parameters to ndarrays
                params = parameters_to_ndarrays(fit_res.parameters)
                
                # Deserialize ensemble
                ensemble = self.deserialize_ensemble(params)
                
                if ensemble:
                    ensembles.append(ensemble)
                    client_weights.append(weights[i])
                    client_metrics.append(fit_res.metrics)
                    
                    # Store client IPFS hash for tracking
                    ipfs_hash = fit_res.metrics.get("client_ipfs_hash", "unknown")
                    client_ipfs_hashes.append(ipfs_hash)
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
        
        # Check if aggregation was successful
        if not aggregated_ensemble:
            logger.error("Ensemble aggregation failed")
            return None, {"error": "ensemble_aggregation_failed"}
        
        # Serialize aggregated ensemble
        parameters = self.serialize_ensemble(aggregated_ensemble)
        
        # Create metrics dictionary
        metrics = {
            "num_ensembles": len(ensembles),
            "num_models": len(aggregated_ensemble["model_names"]),
            "model_names": ",".join(aggregated_ensemble["model_names"]),
            "client_ipfs_hashes": ",".join(client_ipfs_hashes)
        }
        
        # Calculate average GA-Stacking metrics
        ga_metrics = self._aggregate_ga_metrics(client_metrics, client_weights)
        metrics.update(ga_metrics)
        
        # Add ensemble weights to metrics
        for i, (name, weight) in enumerate(zip(aggregated_ensemble["model_names"], aggregated_ensemble["weights"])):
            metrics[f"weight_{name}"] = weight
        
        return ndarrays_to_parameters(parameters), metrics
    
    def _aggregate_ga_metrics(self, client_metrics: List[Dict[str, Any]], weights: List[float]) -> Dict[str, float]:
        """
        Aggregate GA-Stacking metrics from multiple clients.
        
        Args:
            client_metrics: List of metric dictionaries from clients
            weights: Weight for each client
            
        Returns:
            Dictionary of aggregated GA-Stacking metrics
        """
        ga_metric_keys = [
            "ensemble_accuracy", "diversity_score", "generalization_score", 
            "convergence_rate", "final_score"
        ]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0/len(client_metrics)] * len(client_metrics)
        else:
            weights = [w/total_weight for w in weights]
        
        # Initialize aggregated metrics
        aggregated = {}
        
        # Aggregate each GA metric
        for key in ga_metric_keys:
            values = []
            metric_weights = []
            
            for i, metrics in enumerate(client_metrics):
                if key in metrics and isinstance(metrics[key], (int, float)):
                    values.append(float(metrics[key]))
                    metric_weights.append(weights[i])
            
            if values:
                # Normalize weights for this metric
                total = sum(metric_weights)
                if total > 0:
                    normalized_weights = [w/total for w in metric_weights]
                else:
                    normalized_weights = [1.0/len(values)] * len(values)
                
                # Calculate weighted average
                aggregated[f"avg_{key}"] = sum(v * w for v, w in zip(values, normalized_weights))
                aggregated[f"max_{key}"] = max(values)
                aggregated[f"min_{key}"] = min(values)
        
        return aggregated
    
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