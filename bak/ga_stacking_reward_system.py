"""
GA-Stacking Reward System for Federated Learning with Blockchain Integration.
This module handles the evaluation and reward distribution for GA-Stacking ensembles.
"""

import logging
import json
import numpy as np
from web3 import Web3
from datetime import datetime, timezone

class GAStackingRewardSystem:
    """
    A reward system specifically designed for GA-Stacking federated learning.
    Integrates with the blockchain to track and distribute rewards based on
    the quality of GA-Stacking ensembles.
    """
    
    def __init__(self, blockchain_connector, config_path="config/ga_reward_config.json"):
        """
        Initialize the GA-Stacking reward system.
        Adds better error handling for configuration.
        
        Args:
            blockchain_connector: BlockchainConnector instance
            config_path: Path to configuration file
        """
        self.blockchain = blockchain_connector
        self.logger = logging.getLogger('GAStackingRewardSystem')
        self.web3 = blockchain_connector.web3
        
        # Tracking of rewarded clients per round to ensure one-time rewards
        self.rewarded_clients_by_round = {}
        
        # Load GA-specific configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.logger.info(f"Loaded GA-Stacking reward configuration from {config_path}")
            
            # Validate essential configuration keys
            required_keys = [
                "metric_weights",
                "reward_scaling"
            ]
            
            for key in required_keys:
                if key not in self.config:
                    self.logger.warning(f"Missing required config key: {key}, using defaults")
                    # Initialize missing sections with defaults
                    if key == "metric_weights":
                        self.config["metric_weights"] = {
                            "ensemble_accuracy": 0.40,
                            "diversity_score": 0.20,
                            "generalization_score": 0.20,
                            "convergence_rate": 0.10,
                            "avg_base_model_score": 0.10
                        }
                    elif key == "reward_scaling":
                        self.config["reward_scaling"] = {
                            "base_amount": 0.1,
                            "increment_per_round": 0.02,
                            "accuracy_bonus_threshold": 0.9,
                            "bonus_multiplier": 1.5
                        }
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")
            # Set default configuration
            self.config = {
                "metric_weights": {
                    "ensemble_accuracy": 0.40,
                    "diversity_score": 0.20,
                    "generalization_score": 0.20,
                    "convergence_rate": 0.10,
                    "avg_base_model_score": 0.10
                },
                "reward_scaling": {
                    "base_amount": 0.1,
                    "increment_per_round": 0.02,
                    "accuracy_bonus_threshold": 0.9,
                    "bonus_multiplier": 1.5
                }
            }
            self.logger.info("Using default GA-Stacking reward configuration")
    
    def start_training_round(self, round_number):
        """
        Start a new GA-Stacking training round with an appropriate reward pool.
        Only funds if not already funded, does not finalize.
        """
        # Check if pool is already funded
        pool_info = self.get_reward_pool_info(round_number)
        
        if pool_info['total_eth'] > 0:
            self.logger.info(f"Round {round_number} pool already funded with {pool_info['total_eth']} ETH")
            return True, None
        
        # Calculate dynamic reward amount
        base_amount = self.config["reward_scaling"]["base_amount"]
        increment = self.config["reward_scaling"]["increment_per_round"]
        reward_amount = base_amount + (round_number - 1) * increment
        
        # Round to 6 decimal places to avoid floating point issues
        reward_amount = round(reward_amount, 6)
        
        # Fund the pool
        try:
            tx_hash = self.blockchain.fund_round_reward_pool(round_number, reward_amount)
            if tx_hash:
                self.logger.info(f"Successfully funded round {round_number} with {reward_amount} ETH")
                return True, tx_hash
            else:
                self.logger.error(f"Transaction failed for funding round {round_number}")
                return False, None
        except Exception as e:
            self.logger.error(f"Error funding reward pool for round {round_number}: {e}")
            return False, None
    
    def record_client_contribution(self, client_address, ipfs_hash, metrics, round_number):
        """
        Record a client's GA-Stacking contribution on the blockchain.
        Includes duplicate prevention and clear score tracking.
        
        Args:
            client_address: Client's Ethereum address
            ipfs_hash: IPFS hash of the client's model
            metrics: Evaluation metrics dict with GA-Stacking measures
            round_number: Current FL round number
            
        Returns:
            tuple: (success, recorded_score, transaction_hash)
        """
        try:
            # First, verify this client hasn't already been rewarded for this round
            if round_number in self.rewarded_clients_by_round and client_address in self.rewarded_clients_by_round[round_number]:
                self.logger.info(f"Client {client_address} already rewarded for round {round_number}. Skipping.")
                # Return the previously recorded score
                return True, self.rewarded_clients_by_round[round_number][client_address], None
                
            # Check if client already has a higher contribution in this round
            existing_contributions = self.get_round_contributions(round_number)
            client_previous_scores = [
                c['score'] for c in existing_contributions 
                if c['client_address'].lower() == client_address.lower()
            ]
            
            # Ensure we have a valid score
            if 'final_score' not in metrics:
                # Calculate score from individual metrics
                weights = self.config["metric_weights"]
                weighted_score = (
                    metrics.get('ensemble_accuracy', 0.0) * weights['ensemble_accuracy'] +
                    metrics.get('diversity_score', 0.0) * weights['diversity_score'] +
                    metrics.get('generalization_score', 0.0) * weights['generalization_score'] +
                    metrics.get('convergence_rate', 0.5) * weights['convergence_rate'] +
                    metrics.get('avg_base_model_score', 0.0) * weights['avg_base_model_score']
                )
                
                # Apply bonus for exceptional accuracy
                bonus_threshold = self.config["reward_scaling"]["accuracy_bonus_threshold"]
                if metrics.get('ensemble_accuracy', 0.0) > bonus_threshold:
                    bonus_multiplier = self.config["reward_scaling"]["bonus_multiplier"]
                    additional_score = (metrics['ensemble_accuracy'] - bonus_threshold) * bonus_multiplier
                    weighted_score += additional_score
                
                # Convert to integer score (0-10000)
                metrics['final_score'] = int(min(1.0, weighted_score) * 10000)
            
            raw_score = metrics['final_score']
            normalized_score = self.normalize_score(raw_score)
            
            # Skip if client already has a higher score in this round
            if client_previous_scores and max(client_previous_scores) >= raw_score:
                self.logger.info(
                    f"Client {client_address} already has a higher score ({max(client_previous_scores)}) "
                    f"in round {round_number}. Skipping new contribution with score {raw_score}."
                )
                return True, max(client_previous_scores), None
            
            # Record on blockchain
            tx_hash = self.blockchain.contract.functions.recordContribution(
                client_address,
                round_number,
                ipfs_hash,
                normalized_score  # Use normalized score for blockchain (0-100)
            ).transact({
                'from': self.blockchain.account.address,
                'gas': 2000000  # Set reasonable gas limit
            })
            
            # Wait for transaction confirmation
            tx_receipt = self.blockchain.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if tx_receipt.status == 1:
                self.logger.info(
                    f"Recorded GA-Stacking contribution from {client_address} "
                    f"with raw score {raw_score}, normalized to {normalized_score}"
                )
                
                # Track that this client has been processed for this round
                if round_number not in self.rewarded_clients_by_round:
                    self.rewarded_clients_by_round[round_number] = {}
                self.rewarded_clients_by_round[round_number][client_address] = normalized_score
                
                return True, normalized_score, tx_hash.hex()
            else:
                self.logger.error(f"Failed to record GA-Stacking contribution for {client_address}")
                return False, 0, tx_hash.hex()
            
        except Exception as e:
            self.logger.error(f"Error recording GA-Stacking contribution: {e}")
            return False, 0, None
        
    def normalize_score(self, score):
        """
        Normalize score to ensure consistency between recording and allocation.
        Converts from raw scores (0-10000) to display scores (0-100).
        
        Args:
            score: Raw score (0-10000)
            
        Returns:
            normalized_score: Normalized score (0-100)
        """
        # If score is in the 0-10000 range, convert to 0-100 scale
        if score > 100:
            normalized = int(score / 100)
            self.logger.info(f"Normalized score from {score} to {normalized}")
            return normalized
        return score
    
    def finalize_round_and_allocate_rewards(self, round_number):
        """
        Finalize a round and allocate rewards to contributors.
        Handles the case where pool might already be finalized.
        
        Args:
            round_number: Federated learning round number
            
        Returns:
            tuple: (success, allocated_amount)
        """
        try:
            # Check if pool is already finalized
            pool_info = self.get_reward_pool_info(round_number)
            
            # Verify pool has expected amount
            expected_base = self.config["reward_scaling"]["base_amount"]
            expected_increment = self.config["reward_scaling"]["increment_per_round"]
            expected_amount = expected_base + (round_number - 1) * expected_increment
            expected_amount = round(expected_amount, 6)  # Round to 6 decimal places
            
            if abs(float(pool_info['total_eth']) - expected_amount) > 0.0001:
                self.logger.warning(
                    f"Pool amount for round {round_number} ({pool_info['total_eth']} ETH) "
                    f"differs from expected amount ({expected_amount} ETH)"
                )
            
            # If not finalized yet, finalize it
            if not pool_info['is_finalized']:
                self.logger.info(f"Finalizing reward pool for round {round_number}")
                try:
                    tx_hash = self.blockchain.finalize_round_reward_pool(round_number)
                    if not tx_hash:
                        self.logger.error(f"Failed to finalize reward pool for round {round_number}")
                        return False, 0
                    
                    # Wait for transaction confirmation
                    tx_receipt = self.blockchain.web3.eth.wait_for_transaction_receipt(tx_hash)
                    if tx_receipt.status == 1:
                        self.logger.info(f"Finalized reward pool for round {round_number}")
                    else:
                        self.logger.error(f"Transaction to finalize pool for round {round_number} failed")
                        return False, 0
                        
                except Exception as finalize_err:
                    # Check if it's the "already finalized" error
                    error_str = str(finalize_err)
                    if "Already finalized" in error_str:
                        self.logger.info(f"Pool for round {round_number} was already finalized")
                    else:
                        self.logger.error(f"Error finalizing reward pool: {finalize_err}")
                        return False, 0
            else:
                self.logger.info(f"Pool for round {round_number} is already finalized")
            
            # Verify we have contributions for this round before allocating
            contributions = self.get_round_contributions(round_number)
            if not contributions:
                self.logger.warning(f"No contributions found for round {round_number}, allocation will likely fail")
            else:
                self.logger.info(f"Found {len(contributions)} contributions for round {round_number}")
                total_score = sum(c['score'] for c in contributions)
                self.logger.info(f"Total score for round {round_number}: {total_score}")
                if total_score == 0:
                    self.logger.error(f"Total score for round {round_number} is zero, allocation will fail")
                    return False, 0
            
            # Now allocate rewards using the correct contract function
            try:
                tx_hash = self.blockchain.allocate_rewards_for_round(round_number)
                
                if not tx_hash:
                    self.logger.warning(f"Failed to allocate rewards for round {round_number}")
                    return False, 0
                    
                # Get updated pool info after allocation
                updated_pool_info = self.get_reward_pool_info(round_number)
                allocated_eth = updated_pool_info['allocated_eth']
                
                # Get client reward info from contributions
                contributions = self.get_round_contributions(round_number)
                client_summary = {}
                
                for contrib in contributions:
                    client_addr = contrib['client_address']
                    score = contrib['score']
                    # For now, just estimate rewards based on scores
                    if client_addr not in client_summary:
                        client_summary[client_addr] = {
                            'score': score,
                            'estimated_eth': (score / 100) * float(allocated_eth)
                        }
                
                # Log a clear consolidated summary
                self.logger.info(f"=== Round {round_number} Reward Allocation Summary ===")
                self.logger.info(f"Total pool: {updated_pool_info['total_eth']} ETH")
                self.logger.info(f"Allocated: {allocated_eth} ETH")
                self.logger.info(f"Remaining: {updated_pool_info['remaining_eth']} ETH")
                
                # Log client summaries
                self.logger.info(f"=== Client Reward Summary ===")
                for addr, data in client_summary.items():
                    self.logger.info(
                        f"Client {addr} estimated to receive ~{data['estimated_eth']:.6f} ETH "
                        f"in round {round_number} with score: {data['score']}"
                    )
                
                self.logger.info(f"Successfully allocated {allocated_eth} ETH rewards for round {round_number}")
                return True, float(allocated_eth)
                
            except Exception as alloc_err:
                self.logger.error(f"Error allocating rewards: {alloc_err}")
                return False, 0
                
        except Exception as e:
            self.logger.error(f"Error in reward allocation: {e}")
            return False, 0
    
    def get_client_rewards_by_round(self, round_number):
        """
        Get rewards received by clients for a specific round.
        
        Args:
            round_number: Federated learning round number
            
        Returns:
            dict: Dictionary mapping client addresses to reward information
        """
        try:
            # Get contributions for this round
            contributions = self.get_round_contributions(round_number)
            
            # Get client rewards from blockchain events or state
            client_rewards = {}
            
            for contribution in contributions:
                client_address = contribution['client_address']
                score = contribution['score']
                
                # Try to get reward amount from blockchain
                try:
                    # Get the client's contribution info
                    client_info = self.blockchain.contract.functions.getClientContribution(client_address).call()
                    rewards_earned = self.web3.from_wei(client_info[4], 'ether')  # Assuming rewards_earned is at index 4
                    
                    client_rewards[client_address] = {
                        'amount': float(rewards_earned),
                        'score': score
                    }
                except Exception as e:
                    self.logger.warning(f"Could not retrieve reward info for {client_address}: {e}")
                    client_rewards[client_address] = {
                        'amount': 0,
                        'score': score
                    }
            
            return client_rewards
            
        except Exception as e:
            self.logger.error(f"Error getting client rewards by round: {e}")
            return {}
    
    def get_reward_pool_info(self, round_number):
        """
        Get information about a round's reward pool.
        
        Args:
            round_number: The federated learning round number
            
        Returns:
            dict: Reward pool information
        """
        try:
            # Get pool info from blockchain
            pool_info = self.blockchain.contract.functions.getRoundRewardPool(round_number).call()
            total_amount, allocated_amount, remaining_amount, is_finalized = pool_info
            
            # Convert to ETH
            total_eth = self.blockchain.web3.from_wei(total_amount, 'ether')
            allocated_eth = self.blockchain.web3.from_wei(allocated_amount, 'ether')
            remaining_eth = self.blockchain.web3.from_wei(remaining_amount, 'ether')
            
            # Validate - warn about unusual amounts
            try:
                if "reward_scaling" in self.config and "base_amount" in self.config["reward_scaling"]:
                    expected_base = self.config["reward_scaling"]["base_amount"]
                    expected_increment = self.config["reward_scaling"]["increment_per_round"]
                    expected_amount = expected_base + (round_number - 1) * expected_increment
                    expected_amount = round(expected_amount, 6)  # Round to avoid floating point issues
                    
                    if abs(float(total_eth) - expected_amount) > 0.0001:
                        self.logger.warning(
                            f"Unusual pool amount for round {round_number}: {total_eth} ETH "
                            f"(expected around {expected_amount} ETH)"
                        )
            except Exception as config_err:
                self.logger.warning(f"Could not validate pool amount: {config_err}")
            
            return {
                'round': round_number,
                'total_eth': total_eth,
                'allocated_eth': allocated_eth,
                'remaining_eth': remaining_eth,
                'is_finalized': is_finalized
            }
            
        except Exception as e:
            self.logger.error(f"Error getting reward pool info: {e}")
            return {
                'round': round_number,
                'total_eth': 0,
                'allocated_eth': 0,
                'remaining_eth': 0,
                'is_finalized': False
            }
    
    def get_round_contributions(self, round_number, offset=0, limit=100):
        """
        Get all contributions for a specific round with pagination.
        
        Args:
            round_number: The federated learning round number
            offset: Starting index for pagination
            limit: Maximum number of records to return
            
        Returns:
            list: List of contribution records
        """
        try:
            # Get contributions from the contract
            result = self.blockchain.contract.functions.getRoundContributions(
                round_number,
                offset,
                limit
            ).call()
            
            clients, accuracies, scores, rewarded = result
            
            # Format the results as a list of dictionaries
            contributions = []
            for i in range(len(clients)):
                if clients[i] != '0x0000000000000000000000000000000000000000':  # Skip empty entries
                    contributions.append({
                        'client_address': clients[i],
                        'accuracy': accuracies[i] / 10000.0,  # Convert back to percentage
                        'score': scores[i],
                        'rewarded': rewarded[i]
                    })
            
            return contributions
            
        except Exception as e:
            self.logger.error(f"Error getting round contributions: {e}")
            return []
    
    def verify_client_contribution_recorded(self, client_address, round_number):
        """
        Verify that a client's contribution was properly recorded on the blockchain.
        
        Args:
            client_address: Client's Ethereum address
            round_number: The FL round number
            
        Returns:
            tuple: (is_recorded, score)
        """
        try:
            contributions = self.get_round_contributions(round_number)
            
            for contribution in contributions:
                if contribution['client_address'].lower() == client_address.lower():
                    return True, contribution['score']
            
            return False, 0
        except Exception as e:
            self.logger.error(f"Error verifying client contribution: {e}")
            return False, 0
    
    def get_round_contributions_with_metrics(self, round_number):
        """
        Get all contributions for a round with detailed metrics.
        
        Args:
            round_number: Federated learning round number
            
        Returns:
            dict: Detailed contribution records with statistics
        """
        contributions = self.get_round_contributions(round_number)
        
        # Enrich with GA-specific statistics and analysis
        if contributions:
            # Calculate average score
            scores = [c['score'] for c in contributions]
            avg_score = sum(scores) / len(scores)
            
            # Calculate distribution statistics
            score_std = np.std(scores) if len(scores) > 1 else 0
            score_min = min(scores) if scores else 0
            score_max = max(scores) if scores else 0
            
            # Add analysis to each contribution
            for contribution in contributions:
                # Calculate relative performance (percentile)
                contribution['percentile'] = sum(1 for s in scores if s <= contribution['score']) / len(scores)
                
                # Calculate z-score (how many standard deviations from mean)
                if score_std > 0:
                    contribution['z_score'] = (contribution['score'] - avg_score) / score_std
                else:
                    contribution['z_score'] = 0
            
            # Add summary statistics
            contributions_with_stats = {
                'contributions': contributions,
                'summary': {
                    'count': len(contributions),
                    'avg_score': avg_score,
                    'std_deviation': score_std,
                    'min_score': score_min,
                    'max_score': score_max,
                    'score_range': score_max - score_min
                }
            }
            
            return contributions_with_stats
        
        return {'contributions': [], 'summary': {}}
    
    def get_client_rewards(self, client_address):
        """
        Get available rewards for a client.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            float: Available rewards in ETH
        """
        try:
            rewards_wei = self.blockchain.contract.functions.getAvailableRewards(client_address).call()
            rewards_eth = Web3.from_wei(rewards_wei, 'ether')
            return float(rewards_eth)
        except Exception as e:
            self.logger.error(f"Error getting client rewards: {e}")
            return 0.0
        
    def fund_round_reward_pool(self, round_number, amount_eth=None):
        """
        Fund a specific round's reward pool.
        
        Args:
            round_number: Round number
            amount_eth: Amount of ETH to allocate (if None, uses dynamic calculation)
            
        Returns:
            tuple: (success, tx_hash)
        """
        # Calculate dynamic reward amount if not specified
        if amount_eth is None:
            base_amount = self.config["reward_scaling"]["base_amount"]
            increment = self.config["reward_scaling"]["increment_per_round"]
            amount_eth = base_amount + (round_number - 1) * increment
            amount_eth = round(amount_eth, 6)  # Round to 6 decimal places
        
        # Fund the pool
        try:
            tx_hash = self.blockchain.fund_round_reward_pool(round_num=round_number, amount_eth=amount_eth)
            
            if tx_hash:
                self.logger.info(f"Successfully funded round {round_number} with {amount_eth} ETH")
                return True, tx_hash
            else:
                self.logger.error(f"Transaction failed for funding round {round_number}")
                return False, None
        except Exception as e:
            self.logger.error(f"Error funding reward pool for round {round_number}: {e}")
            return False, None
    
    def process_client_for_round(self, client_address, ipfs_hash, metrics, round_number):
        """
        Process a client's contribution for a specific round - one time only.
        This method ensures a client is only rewarded once per round.
        
        Args:
            client_address: Client's Ethereum address
            ipfs_hash: IPFS hash of the client's model
            metrics: Evaluation metrics dict
            round_number: Federated learning round number
            
        Returns:
            tuple: (success, transaction_hash)
        """
        # Check if this client has already been processed for this round
        if round_number in self.rewarded_clients_by_round and client_address in self.rewarded_clients_by_round[round_number]:
            self.logger.info(f"Client {client_address} already processed for round {round_number}. Skipping.")
            return True, None
        
        # Record the contribution
        success, score, tx_hash = self.record_client_contribution(client_address, ipfs_hash, metrics, round_number)
        
        if success:
            # Store that this client has been processed for this round
            if round_number not in self.rewarded_clients_by_round:
                self.rewarded_clients_by_round[round_number] = {}
            self.rewarded_clients_by_round[round_number][client_address] = score
            
            self.logger.info(f"Successfully processed client {client_address} for round {round_number} with score {score}")
            return True, tx_hash
        else:
            self.logger.error(f"Failed to process client {client_address} for round {round_number}")
            return False, None
        
    def log_reward_summary(self, round_number, reward_allocations=None):
        """
        Log a clear consolidated summary of rewards allocated for a round.
        
        Args:
            round_number: FL round number
            reward_allocations: Optional precomputed reward allocations
        """
        try:
            # Get pool info
            pool_info = self.get_reward_pool_info(round_number)
            
            # Get consolidated rewards by client if not provided
            if reward_allocations is None:
                client_rewards = self.get_client_rewards_by_round(round_number)
            else:
                client_rewards = reward_allocations
            
            # Log summary header
            self.logger.info(f"=== Round {round_number} Reward Allocation Summary ===")
            self.logger.info(f"Total pool: {pool_info['total_eth']} ETH")
            self.logger.info(f"Allocated: {pool_info['allocated_eth']} ETH")
            self.logger.info(f"Remaining: {pool_info['remaining_eth']} ETH")
            
            # Log client summaries
            self.logger.info(f"=== Client Reward Summary ===")
            for address, data in client_rewards.items():
                if isinstance(data, dict) and 'amount' in data:
                    # New format
                    self.logger.info(
                        f"Client {address} received {data['amount']:.6f} ETH "
                        f"in round {round_number} with score: {data['score']}"
                    )
                elif isinstance(data, dict) and 'total_rewards' in data:
                    # Old format
                    self.logger.info(
                        f"Client {address} received {data['total_rewards']:.6f} ETH from "
                        f"{data['contributions']} contributions with avg score {data['avg_score']:.2f}"
                    )
        except Exception as e:
            self.logger.error(f"Error generating reward summary: {e}")