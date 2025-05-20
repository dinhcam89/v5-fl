import os
import json
import time
import logging
from web3 import Web3
from web3.middleware import geth_poa_middleware
from decimal import Decimal

class RewardManager:
    """
    Manages the reward process for clients in federated learning.
    Works with the Federation smart contract to track, allocate, and distribute rewards.
    """
    
    def __init__(self, blockchain_connector, config_path="config/blockchain_config.json"):
        """
        Initialize the RewardManager with blockchain connectivity.
        
        Args:
            blockchain_connector: An instance of BlockchainConnector
            config_path: Path to the blockchain configuration file
        """
        self.blockchain_connector = blockchain_connector
        self.logger = logging.getLogger('RewardManager')
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # Set default reward amounts if not specified
            if 'reward_pool_base_amount' not in self.config:
                self.config['reward_pool_base_amount'] = 0.1  # Default 0.1 ETH per round
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {
                'reward_pool_base_amount': 0.1
            }
    
    def fund_reward_pool(self, round_number, amount=None):
        """
        Fund the reward pool for a specific training round.
        
        Args:
            round_number: The federated learning round number
            amount: Amount to fund in ETH (if None, uses the base amount from config)
            
        Returns:
            tuple: (success, transaction_hash)
        """
        try:
            # Use default amount if none specified
            if amount is None:
                amount = self.config.get('reward_pool_base_amount', 0.1)
            
            # Convert ETH to Wei
            amount_wei = Web3.to_wei(amount, 'ether')
            
            # Call the contract function to fund the pool
            tx_hash = self.blockchain_connector.contract.functions.fundRoundRewardPool(
                round_number
            ).transact({
                'from': self.blockchain_connector.account.address,
                'value': amount_wei
            })
            
            # Wait for transaction to be mined
            tx_receipt = self.blockchain_connector.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if tx_receipt.status == 1:
                self.logger.info(f"Successfully funded round {round_number} with {amount} ETH")
                return True, tx_hash.hex()
            else:
                self.logger.error(f"Transaction failed for funding round {round_number}")
                return False, tx_hash.hex()
                
        except Exception as e:
            self.logger.error(f"Error funding reward pool for round {round_number}: {e}")
            return False, None
    
    def finalize_reward_pool(self, round_number):
        """
        Finalize the reward pool for a round, preventing further funding.
        This should be called after a round is complete.
        
        Args:
            round_number: The federated learning round number
            
        Returns:
            tuple: (success, transaction_hash)
        """
        try:
            tx_hash = self.blockchain_connector.contract.functions.finalizeRoundRewardPool(
                round_number
            ).transact({
                'from': self.blockchain_connector.account.address
            })
            
            # Wait for transaction confirmation
            tx_receipt = self.blockchain_connector.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if tx_receipt.status == 1:
                self.logger.info(f"Successfully finalized reward pool for round {round_number}")
                return True, tx_hash.hex()
            else:
                self.logger.error(f"Failed to finalize reward pool for round {round_number}")
                return False, tx_hash.hex()
                
        except Exception as e:
            self.logger.error(f"Error finalizing reward pool for round {round_number}: {e}")
            return False, None
    
    def record_client_contribution(self, client_address, round_number, ipfs_hash, accuracy):
        """
        Record a client's contribution to the federated learning process.
        
        Args:
            client_address: Ethereum address of the client
            round_number: The federated learning round number
            ipfs_hash: IPFS hash of the client's model update
            accuracy: Model accuracy * 10000 (e.g., 95.67% = 9567)
            
        Returns:
            tuple: (success, score, transaction_hash)
        """
        try:
            # Validate input parameters
            if not Web3.is_address(client_address):
                self.logger.error(f"Invalid client address: {client_address}")
                return False, 0, None
            
            if not isinstance(accuracy, int) or not (0 <= accuracy <= 10000):
                self.logger.error(f"Accuracy must be an integer between 0 and 10000: {accuracy}")
                return False, 0, None
            
            # Record the contribution on-chain
            tx_hash = self.blockchain_connector.contract.functions.recordContribution(
                client_address,
                round_number,
                ipfs_hash,
                accuracy
            ).transact({
                'from': self.blockchain_connector.account.address
            })
            
            # Wait for transaction confirmation
            tx_receipt = self.blockchain_connector.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if tx_receipt.status == 1:
                # Get the score from the event logs
                log = self.blockchain_connector.contract.events.ContributionRecorded().process_receipt(tx_receipt)
                score = log[0]['args']['score'] if log else 0
                
                self.logger.info(f"Recorded contribution from {client_address} for round {round_number} with score {score}")
                return True, score, tx_hash.hex()
            else:
                self.logger.error(f"Failed to record contribution for {client_address} in round {round_number}")
                return False, 0, tx_hash.hex()
                
        except Exception as e:
            self.logger.error(f"Error recording contribution: {e}")
            return False, 0, None
    
    def allocate_rewards(self, round_number):
        """
        Allocate rewards to all clients who contributed to a specific round.
        
        Args:
            round_number: The federated learning round number
            
        Returns:
            tuple: (success, transaction_hash)
        """
        try:
            # First check if the reward pool is finalized
            pool_info = self.blockchain_connector.contract.functions.getRoundRewardPool(round_number).call()
            total_amount, allocated_amount, remaining_amount, is_finalized = pool_info
            
            if not is_finalized:
                self.logger.warning(f"Cannot allocate rewards: Reward pool for round {round_number} is not finalized")
                return False, None
            
            if remaining_amount == 0:
                self.logger.warning(f"No remaining rewards to allocate for round {round_number}")
                return False, None
            
            # Allocate rewards
            tx_hash = self.blockchain_connector.contract.functions.allocateRewardsForRound(
                round_number
            ).transact({
                'from': self.blockchain_connector.account.address,
                'gas': 1000000  # Higher gas limit for batch processing
            })
            
            # Wait for transaction confirmation
            tx_receipt = self.blockchain_connector.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if tx_receipt.status == 1:
                # Get allocation events
                logs = self.blockchain_connector.contract.events.RewardAllocated().process_receipt(tx_receipt)
                total_allocated = sum(log['args']['amount'] for log in logs)
                
                self.logger.info(f"Successfully allocated {Web3.from_wei(total_allocated, 'ether')} ETH of rewards for round {round_number}")
                return True, tx_hash.hex()
            else:
                self.logger.error(f"Failed to allocate rewards for round {round_number}")
                return False, tx_hash.hex()
                
        except Exception as e:
            self.logger.error(f"Error allocating rewards: {e}")
            return False, None
    
    def get_client_rewards(self, client_address):
        """
        Get available rewards for a client.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            float: Available rewards in ETH
        """
        try:
            rewards_wei = self.blockchain_connector.contract.functions.getAvailableRewards(client_address).call()
            rewards_eth = Web3.from_wei(rewards_wei, 'ether')
            return float(rewards_eth)
        except Exception as e:
            self.logger.error(f"Error getting client rewards: {e}")
            return 0.0
    
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
            result = self.blockchain_connector.contract.functions.getRoundContributions(
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
    
    def get_reward_pool_info(self, round_number):
        """
        Get information about a round's reward pool.
        
        Args:
            round_number: The federated learning round number
            
        Returns:
            dict: Reward pool information
        """
        try:
            pool_info = self.blockchain_connector.contract.functions.getRoundRewardPool(round_number).call()
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


# Integration with the FL Aggregator

class FLAggregatorWithRewards:
    """
    Extension of the Federated Learning Aggregator with reward functionality.
    """
    
    def __init__(self, base_aggregator, reward_manager):
        """
        Initialize the reward-enabled aggregator.
        
        Args:
            base_aggregator: The base FL aggregator instance
            reward_manager: RewardManager instance
        """
        self.base_aggregator = base_aggregator
        self.reward_manager = reward_manager
        self.logger = logging.getLogger('FLAggregatorWithRewards')
        
        # Configure reward evaluation metrics
        self.performance_metrics = {
            'accuracy': 0.7,   # Weight for accuracy
            'f1_score': 0.15,  # Weight for F1 score
            'loss': 0.15       # Weight for loss (inversely proportional)
        }
    
    def start_round_with_rewards(self, round_number, initial_funding=None):
        """
        Start a new FL round with a funded reward pool.
        
        Args:
            round_number: The federated learning round number
            initial_funding: Initial ETH amount to fund the pool (optional)
            
        Returns:
            bool: Success status
        """
        # First, start the regular FL round
        self.base_aggregator.start_round(round_number)
        
        # Then, fund the reward pool for this round
        success, tx_hash = self.reward_manager.fund_reward_pool(round_number, initial_funding)
        
        return success
    
    def calculate_contribution_score(self, client_metrics):
        """
        Calculate a client's contribution score based on performance metrics.
        
        Args:
            client_metrics: Dictionary containing client's model performance metrics
            
        Returns:
            int: Score between 0-10000 (represents accuracy * 10000)
        """
        # Extract metrics with fallbacks for missing values
        accuracy = client_metrics.get('accuracy', 0)
        f1_score = client_metrics.get('f1_score', 0)
        loss = client_metrics.get('loss', float('inf'))
        
        # Normalize loss (lower is better, so we invert it)
        # Assume reasonable loss range is 0-5, but handle outliers
        normalized_loss = max(0, min(1, 1 - (loss / 5)))
        
        # Calculate weighted score
        weighted_score = (
            accuracy * self.performance_metrics['accuracy'] +
            f1_score * self.performance_metrics['f1_score'] +
            normalized_loss * self.performance_metrics['loss']
        )
        
        # Convert to integer score in range 0-10000
        int_score = int(weighted_score * 10000)
        
        # Ensure bounds
        return max(0, min(10000, int_score))
    
    def process_client_update(self, client_address, model_update, metrics, round_number, ipfs_hash):
        """
        Process a client's model update and record their contribution.
        
        Args:
            client_address: Ethereum address of the client
            model_update: The model update from the client
            metrics: Dictionary of model performance metrics
            round_number: The current FL round number
            ipfs_hash: IPFS hash of the client's model update
            
        Returns:
            tuple: (success, contribution_score)
        """
        # Process the model update with the base aggregator
        self.base_aggregator.process_client_update(client_address, model_update, metrics)
        
        # Calculate contribution score
        score = self.calculate_contribution_score(metrics)
        
        # Record the contribution on-chain
        success, recorded_score, tx_hash = self.reward_manager.record_client_contribution(
            client_address,
            round_number,
            ipfs_hash,
            score
        )
        
        return success, recorded_score
    
    def finalize_round_with_rewards(self, round_number):
        """
        Finalize a FL round, aggregate the model, and finalize the reward pool.
        
        Args:
            round_number: The federated learning round number
            
        Returns:
            tuple: (aggregated_model, success)
        """
        # First, finalize the regular FL round
        aggregated_model = self.base_aggregator.aggregate_and_finalize(round_number)
        
        # Then, finalize the reward pool
        success, tx_hash = self.reward_manager.finalize_reward_pool(round_number)
        
        # Allocate rewards
        if success:
            reward_success, reward_tx = self.reward_manager.allocate_rewards(round_number)
            if not reward_success:
                self.logger.warning(f"Failed to allocate rewards for round {round_number}")
        
        return aggregated_model, success
    
    def get_round_statistics(self, round_number):
        """
        Get statistics for a completed round including contribution and reward data.
        
        Args:
            round_number: The federated learning round number
            
        Returns:
            dict: Round statistics
        """
        # Get base statistics
        base_stats = self.base_aggregator.get_round_statistics(round_number)
        
        # Get reward pool info
        reward_info = self.reward_manager.get_reward_pool_info(round_number)
        
        # Get contributions
        contributions = self.reward_manager.get_round_contributions(round_number, 0, 100)
        
        # Combine the information
        stats = {
            **base_stats,
            'reward_pool': reward_info,
            'contributions': contributions,
            'total_participants': len(contributions),
            'avg_contribution_score': sum(c['score'] for c in contributions) / max(1, len(contributions)),
        }
        
        return stats