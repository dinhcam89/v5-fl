"""
GA-Stacking Reward System for Federated Learning with Blockchain Integration.
This module handles the evaluation and reward distribution for GA-Stacking ensembles.
"""

import logging
import json
import numpy as np
from web3 import Web3
from datetime import datetime, timezone
from blockchain_connector import BlockchainConnector

class GAStackingRewardSystem:
    """
    A reward system specifically designed for GA-Stacking federated learning.
    Integrates with the blockchain to track and distribute rewards based on
    the quality of GA-Stacking ensembles.
    """
    
    def __init__(self, blockchain_connector, config_path="config/ga_reward_config.json"):
        """
        Initialize the GA-Stacking reward system.
        
        Args:
            blockchain_connector: BlockchainConnector instance
            config_path: Path to configuration file
        """
        self.blockchain = blockchain_connector
        self.logger = logging.getLogger('GAStackingRewardSystem')
        
        # Load GA-specific configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.logger.info(f"Loaded GA-Stacking reward configuration from {config_path}")
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")
            # Default configuration
            self.config = {
                "metric_weights": {
                    "ensemble_accuracy": 0.40,
                    "diversity_score": 0.20,
                    "generalization_score": 0.20,
                    "convergence_rate": 0.10,
                    "avg_base_model_score": 0.10
                },
                "reward_scaling": {
                    "base_amount": 0.1,  # ETH per round
                    "increment_per_round": 0.02,  # Increase each round
                    "accuracy_bonus_threshold": 0.9,  # Bonus for >90% accuracy
                    "bonus_multiplier": 1.5  # 50% bonus for high accuracy
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
        
        Args:
            client_address: Client's Ethereum address
            ipfs_hash: IPFS hash of the client's model
            metrics: Evaluation metrics dict with GA-Stacking measures
            round_number: Current FL round number
            
        Returns:
            tuple: (success, recorded_score, transaction_hash)
        """
        try:
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
            
            score = metrics['final_score']
            
            # Record on blockchain
            tx_hash = self.blockchain.contract.functions.recordContribution(
                client_address,
                round_number,
                ipfs_hash,
                score
            ).transact({
                'from': self.blockchain.account.address
            })
            
            # Wait for transaction confirmation
            tx_receipt = self.blockchain.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if tx_receipt.status == 1:
                self.logger.info(f"Recorded GA-Stacking contribution from {client_address} with score {score}")
                return True, score, tx_hash.hex()
            else:
                self.logger.error(f"Failed to record GA-Stacking contribution for {client_address}")
                return False, 0, tx_hash.hex()
                
        except Exception as e:
            self.logger.error(f"Error recording GA-Stacking contribution: {e}")
            return False, 0, None
    
    def finalize_round_and_allocate_rewards(self, round_number):
        """
        Finalize a round and allocate rewards to contributors.
        This checks if the pool is already finalized before attempting to finalize it.
        """
        try:
            # Check if pool is already finalized
            pool_info = self.get_reward_pool_info(round_number)
            
            # If not finalized yet, finalize it
            if not pool_info['is_finalized']:
                self.logger.info(f"Finalizing reward pool for round {round_number}")
                tx_hash = self.blockchain.finalize_round_reward_pool(round_number)
                
                if not tx_hash:
                    self.logger.error(f"Failed to finalize reward pool for round {round_number}")
                    return False, 0
                    
                self.logger.info(f"Finalized reward pool for round {round_number}")
            else:
                self.logger.info(f"Pool for round {round_number} is already finalized")
            
            # Now allocate rewards
            self.logger.info(f"Allocating rewards for round {round_number}")
            tx_hash = self.blockchain.allocate_rewards_for_round(round_number)
            
            if tx_hash:
                updated_pool_info = self.get_reward_pool_info(round_number)
                allocated_eth = updated_pool_info['allocated_eth']
                
                self.logger.info(f"Successfully allocated {allocated_eth} ETH rewards for round {round_number}")
                self.log_client_rewards(round_number, tx_hash)
                return True, allocated_eth
            else:
                self.logger.error(f"Failed to allocate rewards for round {round_number}")
                return False, 0
        except Exception as e:
            self.logger.error(f"Error in reward allocation: {e}")
            return False, 0
    
    def get_reward_pool_info(self, round_number):
        """
        Get information about a round's reward pool.
        
        Args:
            round_number: The federated learning round number
            
        Returns:
            dict: Reward pool information
        """
        try:
            pool_info = self.blockchain.contract.functions.getRoundRewardPool(round_number).call()
            total_amount, allocated_amount, remaining_amount, is_finalized = pool_info
            
            return {
                'round': round_number,
                'total_eth': Web3.from_wei(total_amount, 'ether'),
                'allocated_eth': Web3.from_wei(allocated_amount, 'ether'),
                'remaining_eth': Web3.from_wei(remaining_amount, 'ether'),
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
        
    def log_client_rewards(self, round_number, results):
        """
        Log detailed information about rewards allocated to clients.
        
        Args:
            round_number: The federated learning round number
            results: The allocation results from the smart contract
        """
        try:
            # Get all client contributions for this round
            contributions = self.get_round_contributions(round_number)
            
            # Get the reward pool info
            pool_info = self.get_reward_pool_info(round_number)
            
            # Log allocation summary
            self.logger.info(f"=== Round {round_number} Reward Allocation Summary ===")
            self.logger.info(f"Total pool: {pool_info['total_eth']} ETH")
            self.logger.info(f"Allocated: {pool_info['allocated_eth']} ETH")
            self.logger.info(f"Remaining: {pool_info['remaining_eth']} ETH")
            
            # Log individual client rewards
            for contribution in contributions:
                client_address = contribution['client_address']
                score = contribution['score']
                
                # Calculate this client's reward based on contribution proportion
                total_score = sum(c['score'] for c in contributions)
                client_proportion = score / total_score if total_score > 0 else 0
                client_reward = client_proportion * float(pool_info['allocated_eth'])
                
                self.logger.info(f"Client {client_address} received {client_reward:.6f} ETH in round {round_number} with score: {score}")
        except Exception as e:
            self.logger.error(f"Error logging client rewards: {e}")