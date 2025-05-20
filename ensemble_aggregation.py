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
        aggregation_metrics = {
            "num_ensembles": len(ensembles),
            "num_models": len(all_model_names),
            "method": "weighted_average"
        }
        
        return aggregated_ensemble, aggregation_metrics
    
    def aggregate_fit_results(self, authorized_results, weights):
        """
        Aggregate ensemble models from client results with direct parameter exchange.
        
        Args:
            authorized_results: List of (client, fit_res) tuples
            weights: Weights for each result
            
        Returns:
            Tuple of (parameters, metrics)
        """
        # Check if we need to aggregate ensembles
        any_ensemble = False
        for _, fit_res in authorized_results:
            params = parameters_to_ndarrays(fit_res.parameters)
            if len(params) == 1 and params[0].dtype == np.uint8:
                any_ensemble = True
                break
        
        if any_ensemble:
            # Extract ensemble states from parameters
            ensemble_states = []
            client_ids = []
            
            for client, fit_res in authorized_results:
                try:
                    params = parameters_to_ndarrays(fit_res.parameters)
                    if len(params) == 1 and params[0].dtype == np.uint8:
                        ensemble_bytes = params[0].tobytes()
                        ensemble_state = json.loads(ensemble_bytes.decode('utf-8'))
                        ensemble_states.append(ensemble_state)
                        client_ids.append(client.cid)
                        logger.info(f"Successfully extracted ensemble state from client {client.cid}")
                except Exception as e:
                    logger.error(f"Error extracting ensemble state from client {client.cid}: {e}")
            
            if not ensemble_states:
                logger.error("No valid ensemble states found in client parameters")
                return None, {"error": "no_valid_ensembles"}
            
            logger.info(f"Aggregating {len(ensemble_states)} ensemble states from clients: {client_ids}")
            
            try:
                # Aggregate the ensemble states
                aggregated_ensemble = self.aggregate_ensemble_states(ensemble_states, weights[:len(ensemble_states)])
                
                # Create metrics dictionary
                aggregation_metrics = {
                    "method": "weighted_average",
                    "num_ensembles": len(ensemble_states),
                    "num_models": len(aggregated_ensemble.get("model_names", [])) if aggregated_ensemble else 0
                }
                
                # Convert the aggregated ensemble to parameters format
                ensemble_bytes = json.dumps(aggregated_ensemble).encode('utf-8')
                parameters = ndarrays_to_parameters([np.frombuffer(ensemble_bytes, dtype=np.uint8)])
                
                logger.info("Successfully aggregated client ensembles")
                return parameters, aggregation_metrics
            except Exception as e:
                logger.error(f"Error aggregating ensembles: {e}")
                # Include traceback for better debugging
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return None, {"error": "ensemble_aggregation_failed", "message": str(e)}
        else:
            # No ensembles to aggregate
            logger.error("No ensemble parameters found in client results")
            return None, {"error": "no_ensembles_found"}
    
    def fetch_client_models_from_ipfs(self, authorized_results):
        """Fetch client models from IPFS using the hashes in their metrics."""
        models_data = []
        
        for client, fit_res in authorized_results:
            client_ipfs_hash = fit_res.metrics.get("client_ipfs_hash")
            if not client_ipfs_hash:
                logger.warning(f"Client {client.cid} did not provide IPFS hash")
                continue
            
            try:
                # Retrieve model data from IPFS with retry logic
                for attempt in range(3):
                    try:
                        model_data = self.ipfs.get_json(client_ipfs_hash)
                        if model_data:
                            # Attach the model data to the result
                            models_data.append((client, fit_res, model_data))
                            logger.info(f"Successfully retrieved model for client {client.cid} from IPFS: {client_ipfs_hash}")
                            break
                    except Exception as e:
                        logger.warning(f"IPFS retrieval attempt {attempt+1}/3 failed for hash {client_ipfs_hash}: {e}")
                        time.sleep(2)  # Wait before retry
                else:
                    logger.error(f"Failed to retrieve model for client {client.cid} after multiple attempts")
            except Exception as e:
                logger.error(f"Error retrieving model from IPFS for client {client.cid}: {e}")
        
        return models_data
    
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
    
    def aggregate_ensemble_from_ipfs(self, fetched_models, weights):
        """Aggregate ensemble models fetched from IPFS."""
        
        # Extract ensemble states from the fetched models
        ensemble_states = []
        client_ids = []
        
        for client, fit_res, model_data in fetched_models:
            if "ensemble_state" in model_data:
                ensemble_states.append(model_data["ensemble_state"])
                client_ids.append(client.cid)
        
        if not ensemble_states:
            logger.error("No valid ensemble states found in retrieved models")
            return None, {"error": "no_valid_ensembles"}
        
        logger.info(f"Aggregating {len(ensemble_states)} ensemble states from clients: {client_ids}")
        
        try:
            # Aggregate the ensemble states
            aggregated_ensemble, aggregation_metrics = self.aggregate_ensembles(ensemble_states, weights[:len(ensemble_states)])
            
            # Convert the aggregated ensemble to parameters format
            ensemble_bytes = json.dumps(aggregated_ensemble).encode('utf-8')
            parameters = ndarrays_to_parameters([np.frombuffer(ensemble_bytes, dtype=np.uint8)])
            
            logger.info("Successfully aggregated client ensembles")
            return parameters, aggregation_metrics
        except Exception as e:
            logger.error(f"Error aggregating ensembles: {e}")
            # Include traceback for better debugging
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, {"error": "ensemble_aggregation_failed", "message": str(e)}
    
    def aggregate_ensemble_states(self, ensemble_states, weights):
        """
        Aggregate ensemble states directly without IPFS.
        
        Args:
            ensemble_states: List of ensemble state dictionaries
            weights: Weights for each ensemble (usually based on data samples)
            
        Returns:
            Aggregated ensemble state
        """
        logger.info(f"Aggregating {len(ensemble_states)} ensemble states directly")
        
        if not ensemble_states:
            logger.error("No ensemble states to aggregate")
            return None
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            logger.warning("Total weight is zero, using equal weights")
            weights = [1.0/len(ensemble_states)] * len(ensemble_states)
        else:
            weights = [w/total_weight for w in weights]
        
        # Get all unique model names from all ensembles
        all_model_names = set()
        for ensemble in ensemble_states:
            all_model_names.update(ensemble.get("model_names", []))
        
        all_model_names = sorted(all_model_names)
        
        # Aggregate weights for each model
        aggregated_weights = np.zeros(len(all_model_names))
        weight_counts = np.zeros(len(all_model_names))
        
        for i, ensemble in enumerate(ensemble_states):
            ensemble_weight = weights[i]
            ensemble_model_names = ensemble.get("model_names", [])
            ensemble_weights = ensemble.get("weights", [])
            
            # Skip if ensemble is missing required data
            if not ensemble_model_names or not ensemble_weights:
                logger.warning(f"Skipping incomplete ensemble (missing model_names or weights)")
                continue
                
            for j, model_name in enumerate(ensemble_model_names):
                if j < len(ensemble_weights):
                    # Find index in aggregated model names
                    try:
                        agg_idx = all_model_names.index(model_name)
                        # Add weighted contribution
                        aggregated_weights[agg_idx] += ensemble_weights[j] * ensemble_weight
                        weight_counts[agg_idx] += 1
                    except ValueError:
                        logger.warning(f"Model name {model_name} not in aggregated model list")
                        continue
        
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
        
        # Create aggregated ensemble state
        aggregated_ensemble = {
            "model_names": list(all_model_names),
            "weights": aggregated_weights.tolist(),
            "ga_metadata": {
                "num_ensembles_aggregated": len(ensemble_states),
                "client_weights": weights
            }
        }
        
        logger.info(f"Successfully aggregated {len(ensemble_states)} ensemble states with {len(all_model_names)} models")
        return aggregated_ensemble
    
    def aggregate_fine_tuned_models(self, fetched_models, weights, server_round):
        """
        Aggregate fine-tuned ensemble models with special handling for later rounds.
        
        Args:
            fetched_models: List of (client, fit_res, model_data) tuples
            weights: Weights for each model
            server_round: Current server round
            
        Returns:
            Tuple of (parameters, metrics)
        """
        logger.info(f"Round {server_round}: Aggregating fine-tuned ensemble models")
        
        # Extract ensemble states for both direct and IPFS modes
        ensemble_states = []
        client_ids = []
        
        for client, fit_res, model_data in fetched_models:
            if hasattr(model_data, "get") and model_data.get("ensemble_state") is not None:
                # IPFS mode - extract from model_data
                ensemble_states.append(model_data["ensemble_state"])
                client_ids.append(client.cid)
            else:
                # Direct mode - extract from parameters
                try:
                    params = parameters_to_ndarrays(fit_res.parameters)
                    if len(params) == 1 and params[0].dtype == np.uint8:
                        ensemble_bytes = params[0].tobytes()
                        ensemble_state = json.loads(ensemble_bytes.decode('utf-8'))
                        ensemble_states.append(ensemble_state)
                        client_ids.append(client.cid)
                except Exception as e:
                    logger.error(f"Error extracting ensemble state from client {client.cid}: {e}")
        
        if not ensemble_states:
            logger.error("No valid ensemble states found for fine-tuning aggregation")
            return None, {"error": "no_valid_ensembles"}
        
        # Apply higher weights to ensembles with better generalization
        adjusted_weights = weights.copy()
        for i, ensemble in enumerate(ensemble_states):
            # Give bonus weight to ensembles with good generalization
            gen_score = ensemble.get("ga_metadata", {}).get("generalization_score", 0.5)
            if gen_score > 0.7:  # Threshold for good generalization
                adjusted_weights[i] *= 1.2  # 20% bonus
        
        # Normalize adjusted weights
        total = sum(adjusted_weights)
        if total > 0:
            adjusted_weights = [w/total for w in adjusted_weights]
        
        logger.info(f"Fine-tuning aggregation with {len(ensemble_states)} models from clients: {client_ids}")
        
        try:
            # Use the same aggregation method but with adjusted weights
            aggregated_ensemble = self.aggregate_ensemble_states(ensemble_states, adjusted_weights)
            
            # Add fine-tuning metadata
            aggregated_ensemble["ga_metadata"]["round"] = server_round
            aggregated_ensemble["ga_metadata"]["is_fine_tuned"] = True
            
            # Create metrics dictionary
            aggregation_metrics = {
                "method": "weighted_average_fine_tuned",
                "num_ensembles": len(ensemble_states),
                "num_models": len(aggregated_ensemble.get("model_names", [])) if aggregated_ensemble else 0,
                "round": server_round
            }
            
            # Convert the aggregated ensemble to parameters format
            ensemble_bytes = json.dumps(aggregated_ensemble).encode('utf-8')
            parameters = ndarrays_to_parameters([np.frombuffer(ensemble_bytes, dtype=np.uint8)])
            
            logger.info(f"Successfully aggregated {len(ensemble_states)} fine-tuned ensembles")
            return parameters, aggregation_metrics
        except Exception as e:
            logger.error(f"Error aggregating fine-tuned ensembles: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, {"error": "fine_tuning_aggregation_failed", "message": str(e)}