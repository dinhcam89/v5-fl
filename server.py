"""
Enhanced Federated Learning Server with GA-Stacking support, IPFS and Blockchain integration.
Supports client authorization, contribution tracking, and ensemble model aggregation.
"""

import os
import json
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
from datetime import datetime, timezone
import logging
import random
from pathlib import Path
import matplotlib.pyplot as plt
import pytz
import hashlib

import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.server import ServerConfig
from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    NDArrays,
    MetricsAggregationFn,
)
import numpy as np

from ipfs_connector import IPFSConnector
from blockchain_connector import BlockchainConnector
from ensemble_aggregation import EnsembleAggregator
from ga_stacking_reward_system import GAStackingRewardSystem
from server_config import fit_config_fn, evaluate_config_fn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("FL-Server")

class EnhancedFedAvgWithGA(fl.server.strategy.FedAvg):
    """Enhanced Federated Averaging strategy with GA-Stacking, IPFS, and blockchain integration."""
    
    def __init__(
        self,
        *args,
        ipfs_connector: Optional[IPFSConnector] = None,
        blockchain_connector: Optional[BlockchainConnector] = None,
        version_prefix: str = "1.0",
        authorized_clients_only: bool = True,
        round_rewards: int = 1000,  # Reward points to distribute each round
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize the Enhanced FedAvg with GA strategy."""
        super().__init__(*args, **kwargs)
        
        # Initialize IPFS connector
        self.ipfs = ipfs_connector if ipfs_connector is not None else IPFSConnector()
        
        # Initialize blockchain connector
        self.blockchain = blockchain_connector
                
        self.base_version_prefix = version_prefix

        # Initialize ensemble aggregator
        # NOTE: Removed device parameter here for PyTorch-free implementation
        self.ensemble_aggregator = EnsembleAggregator()
        
        # Generate session-specific versioning
        timestamp = int(time.time())
        self.session_id = timestamp
        readable_date = datetime.now().strftime("%m%d")
        # Format: MMDD-last4digits of timestamp
        self.session_tag = f"{readable_date}-{str(timestamp)[-4:]}"
        
        # Create the full version prefix (base.session)
        self.version_prefix = f"{self.base_version_prefix}.{self.session_tag}"
        
        logger.info(f"Initialized with version strategy: base={self.base_version_prefix}, session={self.session_tag}")
        
        # Flag to only allow authorized clients
        self.authorized_clients_only = authorized_clients_only
        
        # Rewards per round
        self.round_rewards = round_rewards
        
        # Set of authorized clients from blockchain
        self.authorized_clients: Set[str] = set()
        
        # Client contributions for current round
        self.current_round_contributions = {}
        
        # Metrics storage
        self.metrics_history = []
        
        # Ensemble aggregator
        self.ensemble_aggregator = EnsembleAggregator()
        
        self.reward_system = GAStackingRewardSystem(self.blockchain)
        
        # Load authorized clients from blockchain
        self._load_authorized_clients()
        
        # Set device
        self.device = device
        
        
        num_rounds = kwargs.get('num_rounds', 3)  # Default to 3 rounds if not specified
        if hasattr(self, "blockchain") and self.blockchain:
            self.initialize_reward_pools(num_rounds)
        
        logger.info(f"Initialized EnhancedFedAvgWithGA with IPFS node: {self.ipfs.ipfs_api_url}")
        if self.blockchain:
            logger.info(f"Blockchain integration enabled")
            if self.authorized_clients_only:
                logger.info(f"Only accepting contributions from authorized clients ({len(self.authorized_clients)} loaded)")
    
    def initialize_reward_pools(self, num_rounds):
        """Initialize reward pools for the specified number of rounds."""
        if not self.reward_system:
            return
        try:
            logger.info(f"Initializing reward pools for {num_rounds} rounds")
            
            for round_num in range(1, num_rounds + 1):
                # Calculate reward amount
                base_amount = 0.1
                increment = 0.02
                reward_amount = base_amount + (round_num - 1) * increment
                
                # Fund the pool without finalizing
                tx_hash = self.blockchain.fund_round_reward_pool(round_num, reward_amount)
                
                if tx_hash:
                    logger.info(f"Round {round_num} reward pool initialized with {reward_amount} ETH, tx: {tx_hash}")
                else:
                    logger.warning(f"Failed to initialize round {round_num} reward pool")
                    
        except Exception as e:
            logger.error(f"Error initializing reward pools: {e}")
    
    def _load_authorized_clients(self):
        """Load authorized clients from blockchain."""
        if self.blockchain:
            try:
                clients = self.blockchain.get_all_authorized_clients()
                self.authorized_clients = set(clients)
                logger.info(f"Loaded {len(self.authorized_clients)} authorized clients from blockchain")
            except Exception as e:
                logger.error(f"Failed to load authorized clients: {e}")
    
    def is_client_authorized(self, wallet_address: str) -> bool:
        """Check if a client is authorized."""
        if not self.authorized_clients_only:
            return True
        
        if not wallet_address or wallet_address == "unknown":
            logger.warning(f"Client provided no wallet address")
            return False
        
        # Check local cache first
        if wallet_address in self.authorized_clients:
            return True
        
        # Check blockchain
        if self.blockchain:
            try:
                is_authorized = self.blockchain.is_client_authorized(wallet_address)
                # Update local cache
                if is_authorized:
                    self.authorized_clients.add(wallet_address)
                
                return is_authorized
            except Exception as e:
                logger.error(f"Failed to check client authorization: {e}")
                # Fall back to local cache
                return wallet_address in self.authorized_clients
        
        return False
    
    def get_version(self, round_num: int) -> str:
        """Generate a version string based on round number."""
        return f"{self.version_prefix}.{round_num}"

    def initialize_parameters(self, client_manager):
        # Simplify this to just return empty parameters instead of model configs
        # The clients will initialize their own models
        return ndarrays_to_parameters([np.array([])])
    
    def configure_fit(self, server_round, parameters, client_manager):
        # Remove complex model parameter passing
        # Just send round info and basic config
        config = {
            "server_round": server_round,
            "ga_stacking": True,
            "local_epochs": 5,
            "validation_split": 0.2,
            "task_type": "fraud_detection",
            "pos_weight": 99.0,
            "use_f1_score": True,
            "detection_threshold": 0.3
        }
        
        # Use empty parameters to reduce payload size - clients initialize their own models
        empty_parameters = ndarrays_to_parameters([np.array([])])
        fit_ins = FitIns(empty_parameters, config)
        
        # Sample clients for this round
        clients = client_manager.sample(
            num_clients=self.min_fit_clients, 
            min_num_clients=self.min_available_clients
        )
        
        return [(client, fit_ins) for client in clients]
        
    def _store_raw_parameters_in_ipfs(self, params_ndarrays: List[np.ndarray], server_round: int) -> str:
        """Store raw parameters in IPFS."""
        # Create state dict from weights
        state_dict = {}
        layer_names = ["linear.weight", "linear.bias"]  # Adjust based on your model
        
        for i, name in enumerate(layer_names):
            if i < len(params_ndarrays):
                state_dict[name] = params_ndarrays[i].tolist()
        
        # Create metadata
        model_metadata = {
            "state_dict": state_dict,
            "info": {
                "round": server_round,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": self.get_version(server_round),
                "is_ensemble": False
            }
        }
        
        # Store in IPFS
        ipfs_hash = self.ipfs.add_json(model_metadata)
        logger.info(f"Stored global model in IPFS: {ipfs_hash}")
        
        return ipfs_hash
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from clients."""

        # Filter out unauthorized clients
        authorized_results = []
        unauthorized_clients = []

        # Initialize GA-Stacking reward system if not already done
        if not hasattr(self, "reward_system") and hasattr(self, "blockchain"):
            # Fund the reward pool for this round
            try:
                reward_amount = 0.1 + (server_round - 1) * 0.02  # Increase rewards with rounds
                success, tx_hash = self.reward_system.start_training_round(server_round)
                if success:
                    logger.info(f"Funded reward pool for round {server_round} with {reward_amount} ETH")
                else:
                    logger.warning(f"Failed to fund reward pool for round {server_round}")
            except Exception as e:
                logger.error(f"Error funding reward pool: {e}")

        for client, fit_res in results:
            wallet_address = fit_res.metrics.get("wallet_address", "unknown")

            # Check for auth error flag
            if fit_res.metrics.get("error") == "client_not_authorized":
                logger.warning(f"Client {wallet_address} reported as unauthorized")
                unauthorized_clients.append((client, wallet_address))
                continue

            # Verify client authorization
            if self.is_client_authorized(wallet_address):
                authorized_results.append((client, fit_res))

                # Record contribution metrics for GA-Stacking rewards
                client_ipfs_hash = fit_res.metrics.get("client_ipfs_hash")
                
                # Collect GA-Stacking metrics for reward calculation
                ga_metrics = {
                    "ipfs_hash": client_ipfs_hash,
                    "ensemble_accuracy": fit_res.metrics.get("ensemble_accuracy", fit_res.metrics.get("accuracy", 0.0)),
                    "diversity_score": fit_res.metrics.get("diversity_score", 0.0),
                    "generalization_score": fit_res.metrics.get("generalization_score", 0.0),
                    "convergence_rate": fit_res.metrics.get("convergence_rate", 0.5),
                    "avg_base_model_score": fit_res.metrics.get("avg_base_model_score", 0.0),
                    "final_score": fit_res.metrics.get("final_score", 0)
                }
                
                # Store contribution metrics for this round
                if client_ipfs_hash and wallet_address != "unknown":
                    self.current_round_contributions[wallet_address] = ga_metrics
            else:
                logger.warning(f"Rejecting contribution from unauthorized client: {wallet_address}")
                unauthorized_clients.append((client, wallet_address))

        # Check if enough clients returned results
        if not authorized_results:
            if unauthorized_clients:
                logger.error(f"All {len(unauthorized_clients)} clients were unauthorized. No aggregation possible.")
            else:
                logger.error("No clients returned results. No aggregation possible.")
            return None, {"error": "no_authorized_clients"}

        # Calculate the total number of examples used for training
        num_examples_total = sum([fit_res.num_examples for _, fit_res in authorized_results])

        # Create weights for weighted average of client models
        if num_examples_total > 0:
            weights = [fit_res.num_examples / num_examples_total for _, fit_res in authorized_results]
        else:
            weights = [1.0 / len(authorized_results) for _ in authorized_results]

        # Check if we need to aggregate ensembles or regular models
        any_ensemble = False
        for _, fit_res in authorized_results:
            params = parameters_to_ndarrays(fit_res.parameters)
            if len(params) == 1 and params[0].dtype == np.uint8:
                any_ensemble = True
                break

        # Aggregate the updates
        if any_ensemble:
            # Use ensemble aggregation
            logger.info("Aggregating ensemble models")
            parameters_aggregated, agg_metrics = self.ensemble_aggregator.aggregate_fit_results(
                authorized_results, weights
            )
        else:
            # Fall back to standard FedAvg
            logger.info("Aggregating standard models")
            parameters_aggregated, metrics = super().aggregate_fit(server_round, authorized_results, failures)
            agg_metrics = metrics

        if parameters_aggregated is not None:
            # Add metrics about client participation
            agg_metrics["total_clients"] = len(results)
            agg_metrics["authorized_clients"] = len(authorized_results)
            agg_metrics["unauthorized_clients"] = len(unauthorized_clients)

            # Store metrics for history
            self.metrics_history.append({
                "round": server_round,
                "metrics": agg_metrics,
                "num_clients": len(authorized_results),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Process client contributions with GA-Stacking reward system
            if hasattr(self, "reward_system") and self.current_round_contributions:
                logger.info(f"Processing {len(self.current_round_contributions)} client contributions with GA-Stacking rewards")

                for wallet_address, ga_metrics in self.current_round_contributions.items():
                    try:
                        # Record contribution with detailed GA-Stacking metrics
                        success, score, tx_hash = self.reward_system.record_client_contribution(
                            client_address=wallet_address,
                            ipfs_hash=ga_metrics["ipfs_hash"],
                            metrics=ga_metrics,
                            round_number=server_round
                        )
                        
                        if success:
                            logger.info(f"Recorded GA-Stacking contribution for {wallet_address} with score {score}, tx: {tx_hash}")
                        else:
                            logger.warning(f"Failed to record GA-Stacking contribution for {wallet_address}")
                    except Exception as e:
                        logger.error(f"Error recording GA-Stacking contribution for {wallet_address}: {e}")

                # Finalize the round and allocate rewards
                try:
                    success, allocated_amount = self.reward_system.finalize_round_and_allocate_rewards(server_round)
                    if success:
                        logger.info(f"Allocated {allocated_amount} ETH rewards for round {server_round}")
                    else:
                        logger.warning(f"Failed to allocate rewards for round {server_round}")
                except Exception as e:
                    logger.error(f"Error allocating rewards for round {server_round}: {e}")

            # Update participating clients in blockchain if available
            if self.blockchain:
                try:
                    # Get the global model hash from the first client's config
                    # Assumes all clients received the same model
                    ipfs_hash = authorized_results[0][1].metrics.get("ipfs_hash", None)

                    if ipfs_hash:
                        # Update model in blockchain with actual client count
                        tx_hash = self.blockchain.register_or_update_model(
                            ipfs_hash=ipfs_hash,
                            round_num=server_round,
                            version=self.get_version(server_round),
                            participating_clients=len(authorized_results)
                        )
                        logger.info(f"Updated model in blockchain with {len(authorized_results)} clients, tx: {tx_hash}")
                except Exception as e:
                    logger.error(f"Failed to update model in blockchain: {e}")

            # Add GA-Stacking reward metrics to the aggregation metrics
            if hasattr(self, "reward_system"):
                try:
                    # Get detailed contribution metrics for this round
                    contributions_data = self.reward_system.get_round_contributions_with_metrics(server_round)
                    if contributions_data and "summary" in contributions_data:
                        summary = contributions_data["summary"]
                        agg_metrics["ga_stacking_avg_score"] = summary.get("avg_score", 0)
                        agg_metrics["ga_stacking_score_range"] = summary.get("score_range", 0)
                        agg_metrics["ga_stacking_reward_participants"] = summary.get("count", 0)
                except Exception as e:
                    logger.error(f"Error getting GA-Stacking contribution metrics: {e}")

        return parameters_aggregated, agg_metrics
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the evaluation round."""
        
        # Convert Parameters object to a list of NumPy arrays
        params_ndarrays = parameters_to_ndarrays(parameters)
        
        # Check if we have an ensemble or a regular model
        is_ensemble = len(params_ndarrays) == 1 and params_ndarrays[0].dtype == np.uint8
        
        # Store in IPFS with evaluation flag
        if is_ensemble:
            # Handle ensemble model
            try:
                # Deserialize ensemble
                ensemble_bytes = params_ndarrays[0].tobytes()
                ensemble_state = json.loads(ensemble_bytes.decode('utf-8'))
                
                # Create metadata with ensemble state and evaluation flag
                model_metadata = {
                    "ensemble_state": ensemble_state,
                    "info": {
                        "round": server_round,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "version": self.get_version(server_round),
                        "is_ensemble": True,
                        "evaluation": True,
                        "num_models": len(ensemble_state["model_names"]),
                        "model_names": ensemble_state["model_names"]
                    }
                }
                
                # Store in IPFS
                ipfs_hash = self.ipfs.add_json(model_metadata)
                logger.info(f"Stored evaluation ensemble model in IPFS: {ipfs_hash}")
                
            except Exception as e:
                logger.error(f"Failed to process ensemble model for evaluation: {e}")
                # Fall back to storing raw parameters
                ipfs_hash = self._store_raw_parameters_in_ipfs_for_eval(params_ndarrays, server_round)
        else:
            # Handle regular model
            ipfs_hash = self._store_raw_parameters_in_ipfs_for_eval(params_ndarrays, server_round)
        
        # Include IPFS hash in config
        config = {"ipfs_hash": ipfs_hash, "server_round": server_round}
        
        # Convert back to Parameters object for EvaluateIns
        evaluate_ins = EvaluateIns(ndarrays_to_parameters(params_ndarrays), config)
        
        # Sample clients for evaluation
        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients, 
            min_num_clients=self.min_available_clients
        )
        
        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
    
    def _store_raw_parameters_in_ipfs_for_eval(self, params_ndarrays: List[np.ndarray], server_round: int) -> str:
        """Store raw parameters in IPFS for evaluation."""
        # Create state dict from weights
        state_dict = {}
        layer_names = ["linear.weight", "linear.bias"]  # Adjust based on your model
        
        for i, name in enumerate(layer_names):
            if i < len(params_ndarrays):
                state_dict[name] = params_ndarrays[i].tolist()
        
        # Create metadata with evaluation flag
        model_metadata = {
            "state_dict": state_dict,
            "info": {
                "round": server_round,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": self.get_version(server_round),
                "is_ensemble": False,
                "evaluation": True
            }
        }
        
        # Store in IPFS
        ipfs_hash = self.ipfs.add_json(model_metadata)
        logger.info(f"Stored evaluation model in IPFS: {ipfs_hash}")
        
        return ipfs_hash
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients with fraud-specific metrics collection."""

        # Filter out unauthorized clients
        authorized_results = []
        unauthorized_clients = []

        for client, eval_res in results:
            wallet_address = eval_res.metrics.get("wallet_address", "unknown")

            # Check for auth error flag
            if eval_res.metrics.get("error") == "client_not_authorized":
                logger.warning(f"Client {wallet_address} reported as unauthorized")
                unauthorized_clients.append((client, wallet_address))
                continue

            # Verify client authorization
            if self.is_client_authorized(wallet_address):
                authorized_results.append((client, eval_res))
            else:
                logger.warning(f"Rejecting evaluation from unauthorized client: {wallet_address}")
                unauthorized_clients.append((client, wallet_address))

        if not authorized_results:
            if unauthorized_clients:
                logger.error(f"All {len(unauthorized_clients)} clients were unauthorized. No evaluation aggregation possible.")
            else:
                logger.error("No clients returned evaluation results.")
            return None, {"error": "no_authorized_clients"}

        # Initialize client metrics dictionary if not existing
        if not hasattr(self, 'client_metrics'):
            self.client_metrics = {}
        
        # Collect client-specific metrics for comparison and visualization
        for client, eval_res in authorized_results:
            wallet_address = eval_res.metrics.get("wallet_address", "unknown")
            client_id = wallet_address if wallet_address != "unknown" else client.cid
            
            # Initialize client entry if needed
            if client_id not in self.client_metrics:
                self.client_metrics[client_id] = []
            
            # Extract basic metrics
            client_round_metrics = {
                "round": server_round,
                "loss": float(eval_res.loss),
                "accuracy": float(eval_res.metrics.get("accuracy", 0)),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Add fraud-specific metrics if available
            for metric in ["precision", "recall", "f1_score", "auc_roc", "specificity"]:
                if metric in eval_res.metrics:
                    client_round_metrics[metric] = float(eval_res.metrics[metric])
            
            # Add confusion matrix components if available
            for metric in ["true_positives", "false_positives", "true_negatives", "false_negatives"]:
                if metric in eval_res.metrics:
                    client_round_metrics[metric] = int(eval_res.metrics[metric])
            
            # Add GA-Stacking metrics if available
            for metric in ["ensemble_accuracy", "diversity_score", "generalization_score", 
                        "convergence_rate", "avg_base_model_score", "final_score"]:
                if metric in eval_res.metrics:
                    client_round_metrics[metric] = float(eval_res.metrics[metric])
            
            # Store metrics for this client
            self.client_metrics[client_id].append(client_round_metrics)
            logger.debug(f"Collected round {server_round} metrics for client {client_id}")

        # Check if any client has returned ensemble metrics
        has_ensemble_metrics = False
        for _, eval_res in authorized_results:
            if eval_res.metrics.get("ensemble_size", 0) > 1:
                has_ensemble_metrics = True
                break

        # Calculate the total number of examples
        num_examples_total = sum([eval_res.num_examples for _, eval_res in authorized_results])

        # Create weights for weighted average
        if num_examples_total > 0:
            weights = [eval_res.num_examples / num_examples_total for _, eval_res in authorized_results]
        else:
            weights = [1.0 / len(authorized_results) for _ in authorized_results]

        # Aggregate evaluation results
        if has_ensemble_metrics:
            # Use ensemble evaluation aggregation
            loss_aggregated, metrics = self.ensemble_aggregator.aggregate_evaluate_results(
                authorized_results, weights
            )
        else:
            # Use standard aggregation
            loss_aggregated, metrics = super().aggregate_evaluate(server_round, authorized_results, failures)

        # Add metrics about client participation
        metrics["total_clients"] = len(results)
        metrics["authorized_clients"] = len(authorized_results)
        metrics["unauthorized_clients"] = len(unauthorized_clients)

        # Aggregate and add fraud detection metrics
        fraud_metrics = ["precision", "recall", "f1_score", "auc_roc", "specificity"]
        for metric in fraud_metrics:
            values = [float(res.metrics.get(metric, 0.0)) for _, res in authorized_results if metric in res.metrics]
            if values:
                # Calculate weighted average for each metric
                metrics[f"avg_{metric}"] = sum(v * w for v, w in zip(values, weights[:len(values)]))
        
        # Aggregate confusion matrix totals across clients
        cm_totals = {
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0
        }
        
        for _, eval_res in authorized_results:
            for metric in cm_totals.keys():
                if metric in eval_res.metrics:
                    cm_totals[metric] += int(eval_res.metrics[metric])
        
        # Add totals to metrics
        for metric, value in cm_totals.items():
            metrics[metric] = value
        
        # Calculate global precision, recall, and F1 from aggregated confusion matrix
        tp = cm_totals["true_positives"]
        fp = cm_totals["false_positives"]
        fn = cm_totals["false_negatives"]
        tn = cm_totals["true_negatives"]
        
        # Calculate global metrics (handle division by zero)
        if tp + fp > 0:
            metrics["global_precision"] = tp / (tp + fp)
        else:
            metrics["global_precision"] = 0.0
            
        if tp + fn > 0:
            metrics["global_recall"] = tp / (tp + fn)
        else:
            metrics["global_recall"] = 0.0
        
        # Calculate global F1
        if metrics["global_precision"] + metrics["global_recall"] > 0:
            p = metrics["global_precision"]
            r = metrics["global_recall"]
            metrics["global_f1"] = 2 * p * r / (p + r)
        else:
            metrics["global_f1"] = 0.0

        # Calculate average of GA-Stacking specific metrics across clients
        ga_metrics_keys = [
            "ensemble_accuracy", 
            "diversity_score", 
            "generalization_score", 
            "convergence_rate", 
            "avg_base_model_score",
            "final_score"
        ]
        
        # Initialize aggregated GA-Stacking metrics
        ga_metrics_avg = {key: 0.0 for key in ga_metrics_keys}
        ga_metrics_count = 0
        
        # Collect GA-Stacking metrics from clients
        for _, eval_res in authorized_results:
            if all(key in eval_res.metrics for key in ga_metrics_keys):
                for key in ga_metrics_keys:
                    ga_metrics_avg[key] += float(eval_res.metrics[key])
                ga_metrics_count += 1
        
        # Calculate averages if any metrics were found
        if ga_metrics_count > 0:
            for key in ga_metrics_keys:
                ga_metrics_avg[key] /= ga_metrics_count
                metrics[f"avg_{key}"] = ga_metrics_avg[key]
            
            # Add additional aggregate metrics
            metrics["ga_clients_reporting"] = ga_metrics_count
        
        # Calculate average accuracy (keeping your existing code)
        accuracies = [res.metrics.get("accuracy", 0.0) for _, res in authorized_results]
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            metrics["avg_accuracy"] = avg_accuracy

        # Add evaluation metrics to history
        if loss_aggregated is not None:
            eval_metrics = {
                "round": server_round,
                "eval_loss": loss_aggregated,
                "eval_metrics": metrics,
                "num_clients": len(authorized_results),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Save evaluation metrics
            self.metrics_history.append(eval_metrics)

            # Log the evaluation results with enhanced fraud metrics
            logger.info(f"Round {server_round} evaluation: Loss={loss_aggregated:.4f}, Accuracy={metrics.get('avg_accuracy', 0):.4f}")
            
            # Log fraud-specific metrics if available
            if 'global_precision' in metrics:
                logger.info(f"Fraud detection metrics: "
                        f"Precision={metrics['global_precision']:.4f}, "
                        f"Recall={metrics['global_recall']:.4f}, "
                        f"F1={metrics['global_f1']:.4f}")
            
            # Log TP/FP/TN/FN if available
            if all(k in metrics for k in ["true_positives", "false_positives", "true_negatives", "false_negatives"]):
                logger.info(f"Confusion matrix: TP={metrics['true_positives']}, "
                        f"FP={metrics['false_positives']}, "
                        f"TN={metrics['true_negatives']}, "
                        f"FN={metrics['false_negatives']}")
            
            # If we have GA-Stacking metrics, log some additional insights (keeping your existing code)
            if ga_metrics_count > 0:
                logger.info(f"GA-Stacking metrics: "
                        f"Ensemble Acc={metrics.get('avg_ensemble_accuracy', 0):.4f}, "
                        f"Diversity={metrics.get('avg_diversity_score', 0):.4f}, "
                        f"Avg Score={metrics.get('avg_final_score', 0):.1f}")
                
                # If we have a reward system, log the reward distribution (keeping your existing code)
                if hasattr(self, "reward_system"):
                    try:
                        reward_info = self.reward_system.get_reward_pool_info(server_round)
                        if reward_info["allocated_eth"] > 0:
                            logger.info(f"Rewards: {reward_info['allocated_eth']:.4f} ETH allocated "
                                    f"({reward_info['allocated_eth'] / max(1, len(authorized_results)):.4f} ETH/client avg)")
                    except Exception as e:
                        logger.error(f"Error getting reward information: {e}")

        return loss_aggregated, metrics

    def save_metrics_history(self, filepath: str = "metrics/metrics_history.json"):
        """Save metrics history to a file with fraud-specific metrics visualizations."""
        # Save combined metrics history
        with open(filepath, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Saved metrics history to {filepath}")
        
        # Save individual round metrics to separate files
        metrics_dir = Path(filepath).parent
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        for round_metrics in self.metrics_history:
            round_num = round_metrics.get("round", 0)
            round_file = metrics_dir / f"round_{round_num}_metrics.json"
            with open(round_file, "w") as f:
                json.dump(round_metrics, f, indent=2)
            logger.info(f"Saved round {round_num} metrics to {round_file}")
        
        # Generate fraud metrics visualizations
        try:
            # Import visualization module
            from metrics_visualization import (
                generate_fraud_metrics_dashboard, 
                plot_metrics_history,
                generate_client_metrics_comparison
            )
            
            # Create visualizations directory
            vis_dir = metrics_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            
            # Generate main dashboard
            dashboard_path = vis_dir / "fraud_metrics_dashboard.png"
            generate_fraud_metrics_dashboard(self.metrics_history, save_path=str(dashboard_path))
            
            # Generate individual metric plots
            metrics_to_plot = [
                "accuracy", "precision", "recall", "f1_score", "auc_roc",
                "true_positives", "false_positives", "false_negatives"
            ]
            
            for metric in metrics_to_plot:
                try:
                    metric_path = vis_dir / f"{metric}_history.png"
                    fig = plot_metrics_history(self.metrics_history, [metric])
                    fig.savefig(metric_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Error creating {metric} plot: {e}")
            
            # Combined plots for related metrics
            try:
                # Precision, Recall, F1
                prec_rec_path = vis_dir / "precision_recall_f1.png"
                fig = plot_metrics_history(self.metrics_history, ["precision", "recall", "f1_score"])
                fig.savefig(prec_rec_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Confusion matrix components
                cm_path = vis_dir / "confusion_matrix_components.png"
                fig = plot_metrics_history(self.metrics_history, ["true_positives", "false_positives", "false_negatives"])
                fig.savefig(cm_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating combined plots: {e}")
            
            # If client data is available, create client comparison plots
            if hasattr(self, 'client_metrics') and self.client_metrics:
                for metric in ["accuracy", "precision", "recall", "f1_score"]:
                    try:
                        comparison_path = vis_dir / f"client_{metric}_comparison.png"
                        fig = generate_client_metrics_comparison(self.client_metrics, metric, save_path=str(comparison_path))
                        plt.close(fig)
                    except Exception as e:
                        logger.error(f"Error creating client comparison for {metric}: {e}")
            
            logger.info(f"Generated fraud metrics visualizations in {vis_dir}")
            
        except ImportError as e:
            logger.error(f"Could not import visualization module: {e}")
        except Exception as e:
            logger.error(f"Error generating fraud metrics visualizations: {e}")
    
    def save_client_stats(self, filepath: str = "metrics/client_stats.json"):
        """Save client contribution statistics to a file."""
        if not self.blockchain:
            logger.warning("Blockchain connector not available. Cannot save client stats.")
            return
        
        client_stats = {}
        metrics_dir = Path(filepath).parent
        
        try:
            # Get all authorized clients
            clients = self.authorized_clients
            
            for client in clients:
                try:
                    # Get contribution details
                    details = self.blockchain.get_client_contribution_details(client)
                    
                    # Get contribution records
                    records = self.blockchain.get_client_contribution_records(client)
                    
                    # Store in stats
                    client_stats[client] = {
                        "details": details,
                        "records": records
                    }
                    
                    # Save individual client stats to separate files
                    client_file = metrics_dir / f"client_{client[-8:]}_stats.json"
                    with open(client_file, "w") as f:
                        json.dump(client_stats[client], f, indent=2)
                    logger.info(f"Saved client {client[-8:]} stats to {client_file}")
                    
                except Exception as e:
                    logger.error(f"Failed to get stats for client {client}: {e}")
            
            # Save combined stats to file
            with open(filepath, "w") as f:
                json.dump(client_stats, f, indent=2)
                
            logger.info(f"Saved combined client stats to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save client stats: {e}")
            
    def save_model_history(self, filepath: str = "metrics/model_history.json"):
        """Save model history from blockchain to a file."""
        if not self.blockchain:
            logger.warning("Blockchain connector not available. Cannot save model history.")
            return
        
        try:
            metrics_dir = Path(filepath).parent
            all_models = []
            
            # Instead of trying to get all models or using version filtering,
            # specifically request models for the rounds we know exist
            # This is more reliable than trying to filter by version
            max_round = len(self.metrics_history)
            logger.info(f"Retrieving models for {max_round} completed rounds")
            
            for round_num in range(1, max_round + 1):
                try:
                    # Try to get the model for this round
                    models = self.blockchain.get_models_by_round(round_num)
                    if models and len(models) > 0:
                        # Get the latest model for this round
                        model_details = self.blockchain.get_latest_model_by_round(round_num)
                        if model_details:
                            all_models.append(model_details)
                            logger.info(f"Found model for round {round_num}")
                            
                            # Try to get the model data from IPFS
                            try:
                                ipfs_hash = model_details.get("ipfs_hash")
                                if ipfs_hash:
                                    model_data = self.ipfs.get_json(ipfs_hash)
                                    if model_data:
                                        # Save the complete model data including weights
                                        model_file = metrics_dir / f"model_round_{round_num}.json"
                                        with open(model_file, "w") as f:
                                            json.dump(model_data, f, indent=2)
                                        logger.info(f"Saved round {round_num} model data to {model_file}")
                                        
                                        # Save a lightweight model info file (without weights)
                                        info_file = metrics_dir / f"model_round_{round_num}_info.json"
                                        model_info = {**model_details}
                                        if model_data and "info" in model_data:
                                            model_info["model_info"] = model_data["info"]
                                        with open(info_file, "w") as f:
                                            json.dump(model_info, f, indent=2)
                            except Exception as e:
                                logger.error(f"Failed to get model data for round {round_num}: {e}")
                                # Save just the model metadata if we couldn't get the full data
                                model_file = metrics_dir / f"model_round_{round_num}_metadata.json"
                                with open(model_file, "w") as f:
                                    json.dump(model_details, f, indent=2)
                except Exception as e:
                    logger.error(f"Error getting model for round {round_num}: {str(e)}")
            
            # Save combined model history
            with open(filepath, "w") as f:
                json.dump(all_models, f, indent=2)
            
            logger.info(f"Saved model history with {len(all_models)} models to {filepath}")
            latest_model = self.blockchain.get_latest_version_model("1.0")
            logger.info(f"Updated Latest model: {latest_model}")
        except Exception as e:
            logger.error(f"Failed to save model history: {e}")
            # Add fallback to save an empty array if all else fails
            with open(filepath, "w") as f:
                json.dump([], f, indent=2)

    # Model versioning strategy
    def get_version_strategy(self, server_round: int) -> dict:
        """
        Generate a comprehensive version strategy with metadata.
        
        Args:
            server_round: Current federated learning round
            
        Returns:
            Dictionary with version string and metadata
        """
        # Generate a session ID based on timestamp if not already set
        if not hasattr(self, 'session_id'):
            # Create a short, human-readable session ID
            timestamp = int(time.time())
            self.session_id = timestamp
            readable_date = datetime.now().strftime("%m%d")
            # Format: MMDD-last4digits of timestamp
            self.session_tag = f"{readable_date}-{str(timestamp)[-4:]}"
            
            # Generate a short hash of training parameters for uniqueness
            config_hash = hashlib.md5(
                f"{self.min_fit_clients}_{self.min_evaluate_clients}_{self.version_prefix}".encode()
            ).hexdigest()[:4]
            
            # Store original version prefix
            self.base_version_prefix = self.version_prefix
            
            # Update version prefix to include session info
            # Format: original_prefix.sessiontag
            self.version_prefix = f"{self.base_version_prefix}.{self.session_tag}"
        
        # Generate full version with round number
        # Format: original_prefix.sessiontag.round
        version = f"{self.version_prefix}.{server_round}"
        
        # Create version metadata
        version_data = {
            "version": version,
            "base_version": self.base_version_prefix,
            "session_id": self.session_id,
            "session_tag": self.session_tag,
            "round": server_round,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        return version_data

    def get_version(self, server_round: int) -> str:
        """
        Generate a version string based on round number (backward compatible).
        
        Args:
            server_round: Current federated learning round
            
        Returns:
            Version string
        """
        version_data = self.get_version_strategy(server_round)
        return version_data["version"]

def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 3,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    fraction_fit: float = 1.0,
    ipfs_url: str = "http://127.0.0.1:5001",
    ganache_url: str = "http://127.0.0.1:7545",
    contract_address: Optional[str] = None,
    private_key: Optional[str] = None,
    deploy_contract: bool = False,
    version_prefix: str = "1.0",
    authorized_clients_only: bool = True,
    authorized_clients: Optional[List[str]] = None,
    round_rewards: int = 1000,
    device: str = "cpu"
) -> None:
    """
    Start the federated learning server.
    
    Args:
        server_address: Server address in format "host:port"
        ipfs_url: IPFS API URL
        ganache_url: Ganache URL
        contract_address: Address of the deployed smart contract
        server_private_key: Private key for the server
        server_wallet_address: Wallet address for the server
        continue_from_round: Round to continue from (0 for new server)
        continue_from_ipfs: IPFS hash to continue from
        output_dir: Directory to save server outputs
        bootstrap_clients: Number of clients to bootstrap with
        min_clients: Minimum number of available clients
        ssl_key: SSL key file path
        ssl_cert: SSL certificate file path
        ca_cert: CA certificate file path
        strategy: Strategy to use ('fedavg_ga' or 'fedavg')
        ga_stacking: Whether to use GA-Stacking
    """
    # Initialize IPFS connector
    ipfs_connector = IPFSConnector(ipfs_api_url=ipfs_url)
    logger.info(f"Initialized IPFS connector: {ipfs_url}")
    
    # Initialize blockchain connector
    blockchain_connector = None
    if ganache_url:
        try:
            blockchain_connector = BlockchainConnector(
                ganache_url=ganache_url,
                contract_address=contract_address,
                private_key=private_key
            )
            
            # Deploy contract if needed
            if contract_address is None and deploy_contract:
                contract_address = blockchain_connector.deploy_contract()
                logger.info(f"Deployed new contract at: {contract_address}")
                
                # Save contract address to file for future use
                with open("contract_address.txt", "w") as f:
                    f.write(contract_address)
                
                # Initialize the blockchain connector with the new contract
                blockchain_connector = BlockchainConnector(
                    ganache_url=ganache_url,
                    contract_address=contract_address,
                    private_key=private_key
                )
            elif contract_address is None:
                logger.warning("No contract address provided and deploy_contract=False. Blockchain features disabled.")
                blockchain_connector = None
                
            # Authorize clients if provided
            if blockchain_connector and authorized_clients:
                # Check which clients are not already authorized
                to_authorize = []
                for client in authorized_clients:
                    if not blockchain_connector.is_client_authorized(client):
                        to_authorize.append(client)
                
                if to_authorize:
                    logger.info(f"Authorizing {len(to_authorize)} new clients")
                    blockchain_connector.authorize_clients(to_authorize)
                else:
                    logger.info("All provided clients are already authorized")
                
        except Exception as e:
            logger.error(f"Failed to initialize blockchain connector: {e}")
            logger.warning("Continuing without blockchain integration")
            blockchain_connector = None
    
    # Configure strategy with GA-Stacking support
    strategy = EnhancedFedAvgWithGA(
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_fit_clients,
        ipfs_connector=ipfs_connector,
        blockchain_connector=blockchain_connector,
        version_prefix=version_prefix,
        authorized_clients_only=authorized_clients_only,
        round_rewards=round_rewards,
        device=device
    )
    
    
    # Create metrics directory with timestamp to keep each training run separate
    vn_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
    local_time = datetime.now(vn_timezone)
    timestamp = local_time.strftime("%Y-%m-%d_%H-%M-%S")
    metrics_dir = Path(f"metrics/run_{timestamp}")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Start server
    server = fl.server.Server(client_manager=fl.server.SimpleClientManager(), strategy=strategy)
    
    # monitor = start_monitoring_server(port=8050)

    # Run server
    fl.server.start_server(
        server_address=server_address,
        server=server,
        config=fl.server.ServerConfig(num_rounds=num_rounds)
    )
    
    # Save metrics history (both combined and per-round)
    strategy.save_metrics_history(filepath=str(metrics_dir / "metrics_history.json"))
    
    # Save client stats
    strategy.save_client_stats(filepath=str(metrics_dir / "client_stats.json"))
    
    # Save model history
    strategy.save_model_history(filepath=str(metrics_dir / "model_history.json"))
    
    # Create a summary file with key information
    summary = {
        "timestamp": timestamp,
        "num_rounds": num_rounds,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "authorized_clients_only": authorized_clients_only,
        "version_prefix": version_prefix,
        "contract_address": contract_address,
        "final_metrics": strategy.metrics_history[-1] if strategy.metrics_history else None
    }
    
    with open(metrics_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Server completed {num_rounds} rounds of federated learning with GA-Stacking")
    logger.info(f"All metrics saved to {metrics_dir}")  

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start enhanced FL server with GA-Stacking, IPFS and blockchain integration")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8088", help="Server address (host:port)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated learning rounds")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum number of clients for training")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum number of clients for evaluation")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients to use for training")
    parser.add_argument("--ipfs-url", type=str, default="http://127.0.0.1:5001/api/v0", help="IPFS API URL")
    parser.add_argument("--ganache-url", type=str, default="http://192.168.1.146:7545", help="Ganache blockchain URL")
    parser.add_argument("--contract-address", type=str, help="Federation contract address")
    parser.add_argument("--private-key", type=str, help="Private key for blockchain transactions")
    parser.add_argument("--deploy-contract", action="store_true", help="Deploy a new contract if address not provided")
    parser.add_argument("--version-prefix", type=str, default="1.0", help="Version prefix for model versioning")
    parser.add_argument("--authorized-clients-only", action="store_true", help="Only accept contributions from authorized clients")
    parser.add_argument("--authorize-clients", nargs="+", help="List of client addresses to authorize")
    parser.add_argument("--round-rewards", type=int, default=1000, help="Reward points to distribute each round")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for computation")
    
    args = parser.parse_args()
    
    # Check if contract address is stored in file
    if args.contract_address is None and not args.deploy_contract:
        try:
            with open("contract_address.txt", "r") as f:
                args.contract_address = f.read().strip()
                logger.info(f"Loaded contract address from file: {args.contract_address}")
        except FileNotFoundError:
            logger.warning("No contract address provided or found in file")
    
    start_server(
        server_address=args.server_address,
        num_rounds=args.rounds,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        fraction_fit=args.fraction_fit,
        ipfs_url=args.ipfs_url,
        ganache_url=args.ganache_url,
        contract_address=args.contract_address,
        private_key=args.private_key,
        deploy_contract=args.deploy_contract,
        version_prefix=args.version_prefix,
        authorized_clients_only=args.authorized_clients_only,
        authorized_clients=args.authorize_clients,
        round_rewards=args.round_rewards,
        device=args.device
    )