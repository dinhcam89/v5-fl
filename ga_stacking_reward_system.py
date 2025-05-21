"""
Simplified GA-Stacking Reward System for Federated Learning with Blockchain Integration.
This module handles the evaluation and reward distribution based on data size, participation, and model quality.
"""

import logging
import json
import numpy as np
from web3 import Web3
from datetime import datetime, timezone
import traceback
import os
import time
import random
from blockchain_connector import BlockchainConnector

class GAStackingRewardSystem:
    """
    A simplified reward system for GA-Stacking federated learning.
    Focuses on data size, training rounds participation, and evaluation results.
    Integrates with the blockchain to track and distribute rewards.
    """
    
    def __init__(self, blockchain_connector, config_path="config/ga_reward_config.json"):
        """
        Initialize the fraud detection focused reward system.
        
        Args:
            blockchain_connector: BlockchainConnector instance
            config_path: Path to configuration file
        """
        self.blockchain = blockchain_connector
        self.logger = logging.getLogger('GAStackingRewardSystem')
        
        # Set default configuration first to ensure it always exists
        # Default configuration with recall and data points focus
        self.config = {
            "metric_weights": {
                "recall": 0.60,         # 60% weight for recall (finding true fraud cases)
                "data_points": 0.40      # 40% weight for data contribution size
            },
            "reward_scaling": {
                "base_amount": 0.1,         # ETH per round
                "increment_per_round": 0.02 # Increase each round
            },
            "data_points_normalization": {
                "min_value": 1000,
                "log_scale": True
            }
        }
        
        # Try to load configuration from file
        try:
            # Check if config file exists
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Update config with loaded values
                    self.config.update(loaded_config)
                self.logger.info(f"Loaded fraud detection reward configuration from {config_path}")
            else:
                self.logger.warning(f"Config file not found at {config_path}, using default configuration")
                
                # Create the config directory if it doesn't exist
                config_dir = os.path.dirname(config_path)
                if not os.path.exists(config_dir):
                    os.makedirs(config_dir)
                
                # Write the default config to file
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=4)
                self.logger.info(f"Created default configuration file at {config_path}")
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")
            self.logger.info("Using default fraud detection reward configuration")
        
    def start_training_round(self, round_number):
        """
        Start a new training round with an appropriate reward pool.
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
            # Check if blockchain connector is properly initialized
            if not self.blockchain:
                self.logger.error("Blockchain connector not initialized")
                return False, None
            
            tx_hash = self.blockchain.fund_round_reward_pool(round_number, reward_amount)
            if tx_hash:
                self.logger.info(f"Successfully funded round {round_number} with {reward_amount} ETH")
                return True, tx_hash
            else:
                self.logger.error(f"Transaction failed for funding round {round_number}")
                return False, None
        except Exception as e:
            self.logger.error(f"Error funding reward pool for round {round_number}: {e}")
            traceback.print_exc()
            return False, None
    
    def record_client_contribution(self, client_address, ipfs_hash, metrics, round_number):
        """
        Record a client's contribution with scoring based on:
        1. Recall - ability to detect true fraud cases
        2. Data points - amount of data contributed
        """
        try:
            self.logger.info(f"Processing fraud detection contribution for {client_address}")
            
            # Extract the recall and data points metrics
            recall = metrics.get('recall', 0)  # Recall from validation/test set
            data_points = metrics.get('data_points', metrics.get('data_size', 0))  # Support both names
            
            # Log the extracted metrics
            self.logger.info(f"Client metrics - recall: {recall}, data_points: {data_points}")
            
            # Get weights from config, using fraud detection defaults if not present
            weights = self.config.get("metric_weights", {
                "recall": 0.60,
                "data_points": 0.40
            })
            
            # Normalize metrics
            try:
                # For recall, ensure it's between 0-1
                recall_float = float(recall)
                norm_recall = max(0.0, min(1.0, recall_float))
            except (TypeError, ValueError):
                norm_recall = 0.1  # Default if recall is invalid
                
            try:
                # For data points, use logarithmic scaling for large datasets
                if data_points <= 0:
                    norm_data_points = 0.0
                elif data_points < 1000:
                    # Linear scaling for small datasets (0-1000 points)
                    norm_data_points = min(1.0, data_points / 1000.0)
                else:
                    # Logarithmic scaling for large datasets (1000+ points)
                    # log10(1000)=3, log10(1M)=6, gives a value between 0-1
                    import math
                    norm_data_points = min(1.0, (math.log10(data_points) - 3) / 3)
            except (TypeError, ValueError):
                norm_data_points = 0.1  # Default if data_points is invalid
            
            # Calculate weighted score
            weighted_score = (
                norm_recall * weights.get('recall', 0.6) +
                norm_data_points * weights.get('data_points', 0.4)
            )
            
            # Apply high recall bonus if configured
            recall_bonus = 0.0
            if self.config.get("recall_bonus", {}).get("enabled", False):
                bonus_config = self.config.get("recall_bonus", {})
                threshold = bonus_config.get("threshold", 0.85)
                multiplier = bonus_config.get("bonus_multiplier", 1.2)
                
                if norm_recall > threshold:
                    recall_bonus = (norm_recall - threshold) * multiplier
                    self.logger.info(f"Applied recall bonus of {recall_bonus:.4f} for exceptional recall")
            
            # Add the bonus to the weighted score
            weighted_score += recall_bonus
            
            # Ensure minimum score
            weighted_score = max(0.01, weighted_score)
            
            # Convert to integer score (1-10000)
            final_score = max(1, int(weighted_score * 10000))
            
            # Log detailed score breakdown
            self.logger.info(f"Calculated fraud detection score for {client_address}: {final_score}")
            self.logger.info(f"  - Recall contribution: {norm_recall * weights.get('recall', 0.6) * 10000:.0f}")
            self.logger.info(f"  - Data points contribution: {norm_data_points * weights.get('data_points', 0.4) * 10000:.0f}")
            if recall_bonus > 0:
                self.logger.info(f"  - Recall bonus: {recall_bonus * 10000:.0f}")
            
            # Check if blockchain connector is properly initialized
            if not self.blockchain:
                self.logger.error("Blockchain connector not initialized")
                return False, 0, None
            
            if not hasattr(self.blockchain, 'contract') or self.blockchain.contract is None:
                self.logger.error("Blockchain contract not initialized")
                return False, 0, None
                
            # Record on blockchain
            try:
                tx_hash = self.blockchain.contract.functions.recordContribution(
                    client_address,
                    round_number,
                    ipfs_hash or "ipfs_placeholder",  # Use placeholder if hash is None
                    final_score
                ).transact({
                    'from': self.blockchain.account.address
                })
                
                # Wait for transaction confirmation
                tx_receipt = self.blockchain.web3.eth.wait_for_transaction_receipt(tx_hash)
                
                if tx_receipt.status == 1:
                    self.logger.info(f"Recorded contribution from {client_address} with score {final_score}")
                    return True, final_score, tx_hash.hex()
                else:
                    self.logger.error(f"Failed to record contribution for {client_address}")
                    return False, 0, tx_hash.hex()
            except AttributeError as e:
                self.logger.error(f"Blockchain interface error: {e}")
                self.logger.error("Check if blockchain connector and contract are properly initialized")
                return False, 0, None
            except Exception as e:
                self.logger.error(f"Transaction error: {e}")
                traceback.print_exc()
                return False, 0, None
                
        except Exception as e:
            self.logger.error(f"Error recording fraud detection contribution: {e}")
            traceback.print_exc()
            return False, 0, None
    
    def finalize_round_and_allocate_rewards(self, round_number):
        """
        Finalize a round and allocate rewards, ensuring only active clients receive rewards
        and expected values match actual blockchain transfers.
        """
        try:
            # Check blockchain connectivity
            if not self.blockchain or not hasattr(self.blockchain, 'contract') or self.blockchain.contract is None:
                self.logger.error("Blockchain contract not initialized")
                return False, 0
            
            # Get round-specific contributions (only active clients)
            contributions = self.get_round_contributions(round_number)
            if not contributions:
                self.logger.warning(f"No contributions found for round {round_number}")
                return False, 0
                
            # Calculate total score for only this round's active participants
            total_score = sum(c['score'] for c in contributions)
            self.logger.info(f"Total score for round {round_number}: {total_score}")
            
            # Handle zero score situation (similar to your existing code)
            if total_score == 0:
                # Your existing zero-score handling code...
                pass
            
            # Get active client addresses from this round for balance tracking
            active_clients = [c['client_address'] for c in contributions]
            
            # Track wallet balances ONLY for active clients
            balance_tracker = self.track_wallet_changes(round_number, active_clients)
            
            # Check if pool is already finalized
            pool_info = self.get_reward_pool_info(round_number)
            
            # Finalize if needed
            if not pool_info['is_finalized']:
                self.logger.info(f"Finalizing reward pool for round {round_number}")
                tx_hash = self.blockchain.finalize_round_reward_pool(round_number)
                if not tx_hash:
                    self.logger.error(f"Failed to finalize reward pool for round {round_number}")
                    return False, 0
            else:
                self.logger.info(f"Pool for round {round_number} is already finalized")
            
            # Calculate expected rewards for ONLY this round
            expected_rewards = self.calculate_round_specific_rewards(round_number)
            
            # Now allocate rewards with retry mechanism
            for attempt in range(3):  # Try up to 3 times
                self.logger.info(f"Allocating rewards for round {round_number} - attempt {attempt+1}/3")
                try:
                    tx_hash = self.blockchain.allocate_rewards_for_round(round_number)
                    
                    if tx_hash:
                        updated_pool_info = self.get_reward_pool_info(round_number)
                        allocated_eth = updated_pool_info['allocated_eth']
                        
                        self.logger.info(f"Successfully allocated {allocated_eth} ETH rewards for round {round_number}")
                        
                        # Record wallet balances after allocation
                        balance_changes = balance_tracker()
                        
                        # Log and verify rewards using our NEW round-specific values
                        self.logger.info(f"=== Reward Verification Summary ===")
                        for client_address, data in expected_rewards.items():
                            expected_reward = data['reward']
                            actual_change = 0
                            
                            # Get actual change from balance tracking
                            if balance_changes and 'changes' in balance_changes and client_address in balance_changes['changes']:
                                actual_change = balance_changes['changes'][client_address]['change']
                            
                            # Compare expected vs actual (allow small difference for gas)
                            match = abs(expected_reward - actual_change) < 0.001
                            status = "✓ Match" if match else "❌ Mismatch"
                            
                            self.logger.info(
                                f"Client {client_address}: Expected {expected_reward:.6f} ETH, "
                                f"Actual change {actual_change:.6f} ETH - {status}"
                            )
                        
                        # Call log_client_rewards for historical tracking purposes
                        self.log_client_rewards(round_number, tx_hash)
                        
                        return True, allocated_eth
                    else:
                        self.logger.warning(f"Reward allocation attempt {attempt+1} failed, retrying...")
                        time.sleep(2)  # Wait before retry
                except Exception as alloc_error:
                    self.logger.error(f"Error during allocation attempt {attempt+1}: {alloc_error}")
                    time.sleep(2)  # Wait before retry
            
            self.logger.error(f"All allocation attempts failed for round {round_number}")
            return False, 0
        except Exception as e:
            self.logger.error(f"Error in reward allocation: {e}")
            self.logger.error(traceback.format_exc())
            return False, 0
    
    def track_wallet_changes(self, round_number, clients_to_track=None):
        """
        Track wallet balance changes before and after reward distribution.
        
        Args:
            round_number: The federated learning round number
            clients_to_track: List of client addresses to track (if None, all authorized clients are tracked)
            
        Returns:
            Dictionary with balance changes
        """
        try:
            # If no specific clients provided, get contributors from this round only
            if clients_to_track is None:
                try:
                    contributions = self.get_round_contributions(round_number)
                    clients_to_track = [c['client_address'] for c in contributions]
                    self.logger.info(f"Tracking {len(clients_to_track)} active clients from round {round_number}")
                except Exception as e:
                    self.logger.error(f"Error getting authorized clients: {e}")
                    # Fallback to current round contributions
                    clients_to_track = [c['client_address'] for c in self.get_round_contributions(round_number)]
                    self.logger.info(f"Fallback: Tracking {len(clients_to_track)} clients from current round")
            
            # Create tracking directory if it doesn't exist
            os.makedirs("metrics/wallet_tracking", exist_ok=True)
            
            # Function to get current balances
            def get_balances(client_addresses):
                balances = {}
                for address in client_addresses:
                    try:
                        # Get ETH balance in wei
                        balance_wei = self.blockchain.web3.eth.get_balance(address)
                        # Convert to ETH
                        balance_eth = self.blockchain.web3.from_wei(balance_wei, 'ether')
                        balances[address] = float(balance_eth)
                    except Exception as e:
                        self.logger.error(f"Error getting balance for {address}: {e}")
                        balances[address] = None
                return balances
            
            # Get balances before allocation
            self.logger.info(f"Recording wallet balances BEFORE reward allocation for round {round_number}")
            before_balances = get_balances(clients_to_track)
            
            # Save pre-allocation balances
            before_file = f"metrics/wallet_tracking/round_{round_number}_before_allocation.json"
            with open(before_file, 'w') as f:
                json.dump(before_balances, f, indent=2)
            self.logger.info(f"Saved pre-allocation balances to {before_file}")
            
            # Return the function that should be called after allocation
            def record_after_allocation():
                # Get balances after allocation
                self.logger.info(f"Recording wallet balances AFTER reward allocation for round {round_number}")
                after_balances = get_balances(clients_to_track)
                
                # Calculate differences
                balance_changes = {}
                for address in before_balances:
                    if before_balances[address] is not None and after_balances.get(address) is not None:
                        # Calculate the difference
                        change = after_balances[address] - before_balances[address]
                        balance_changes[address] = {
                            'before': before_balances[address],
                            'after': after_balances[address],
                            'change': change
                        }
                
                # Log the changes
                self.logger.info(f"=== Wallet Balance Changes for Round {round_number} ===")
                for address, data in balance_changes.items():
                    if abs(data['change']) > 0.000001:  # Only show non-zero changes
                        self.logger.info(f"Client {address}: {data['before']:.6f} ETH → {data['after']:.6f} ETH (Change: {data['change']:.6f} ETH)")
                
                # Save the data
                changes_file = f"metrics/wallet_tracking/round_{round_number}_balance_changes.json"
                result_data = {
                    'before': before_balances,
                    'after': after_balances,
                    'changes': balance_changes
                }
                with open(changes_file, 'w') as f:
                    json.dump(result_data, f, indent=2)
                self.logger.info(f"Saved wallet balance changes to {changes_file}")
                
                return result_data
            
            return record_after_allocation
        
        except Exception as e:
            self.logger.error(f"Error tracking wallet changes: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return a dummy function that does nothing
            return lambda: None
    
    def get_reward_pool_info(self, round_number):
        """
        Get information about a round's reward pool.
        
        Args:
            round_number: The federated learning round number
            
        Returns:
            dict: Reward pool information
        """
        try:
            # Check if blockchain connector is properly initialized
            if not self.blockchain:
                self.logger.error("Blockchain connector not initialized")
                return {
                    'round': round_number,
                    'total_eth': 0,
                    'allocated_eth': 0,
                    'remaining_eth': 0,
                    'is_finalized': False
                }
                
            # Check if contract is available
            if not hasattr(self.blockchain, 'contract') or self.blockchain.contract is None:
                self.logger.error("Blockchain contract not initialized")
                return {
                    'round': round_number,
                    'total_eth': 0,
                    'allocated_eth': 0,
                    'remaining_eth': 0,
                    'is_finalized': False
                }
            
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
            # Check if blockchain connector is properly initialized
            if not self.blockchain:
                self.logger.error("Blockchain connector not initialized")
                return []
                
            # Check if contract is available
            if not hasattr(self.blockchain, 'contract') or self.blockchain.contract is None:
                self.logger.error("Blockchain contract not initialized")
                return []
                
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
            
        except AttributeError as e:
            self.logger.error(f"Blockchain interface error: {e}")
            self.logger.error("Check if blockchain connector and contract are properly initialized")
            return []
        except Exception as e:
            self.logger.error(f"Error getting round contributions: {e}")
            traceback.print_exc()
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
        
        # Enrich with statistics and analysis
        if contributions:
            # Calculate average score
            scores = [c['score'] for c in contributions]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Calculate distribution statistics
            score_std = np.std(scores) if len(scores) > 1 else 0
            score_min = min(scores) if scores else 0
            score_max = max(scores) if scores else 0
            
            # Add analysis to each contribution
            for contribution in contributions:
                # Calculate relative performance (percentile)
                contribution['percentile'] = sum(1 for s in scores if s <= contribution['score']) / len(scores) if scores else 0
                
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
    
    def get_client_contribution_summary(self, client_address):
        """
        Get a summary of a client's contributions across all rounds.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            dict: Summary of client's contributions
        """
        try:
            # Check if blockchain connector is properly initialized
            if not self.blockchain:
                self.logger.error("Blockchain connector not initialized")
                return {
                    'address': client_address,
                    'contribution_count': 0,
                    'total_score': 0,
                    'is_authorized': False,
                    'last_contribution': None,
                    'rewards_earned_eth': 0,
                    'rewards_claimed': False
                }
            
            client_info = self.blockchain.contract.functions.getClientContribution(client_address).call()
            
            # Extract basic info
            contribution_count, total_score, is_authorized, last_timestamp, rewards_earned, rewards_claimed = client_info
            
            # Format into a summary
            summary = {
                'address': client_address,
                'contribution_count': contribution_count,
                'total_score': total_score,
                'is_authorized': is_authorized,
                'last_contribution': datetime.fromtimestamp(last_timestamp, tz=timezone.utc).isoformat() if last_timestamp > 0 else None,
                'rewards_earned_eth': Web3.from_wei(rewards_earned, 'ether'),
                'rewards_claimed': rewards_claimed
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting client contribution summary: {e}")
            return {
                'address': client_address,
                'contribution_count': 0,
                'total_score': 0,
                'is_authorized': False,
                'last_contribution': None,
                'rewards_earned_eth': 0,
                'rewards_claimed': False
            }
    
    def get_client_rewards(self, client_address):
        """
        Get available rewards for a client.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            float: Available rewards in ETH
        """
        try:
            # Check if blockchain connector is properly initialized
            if not self.blockchain:
                self.logger.error("Blockchain connector not initialized")
                return 0.0
                
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
            # Check if blockchain connector is properly initialized
            if not self.blockchain:
                self.logger.error("Blockchain connector not initialized")
                return False, None
                
            tx_hash = self.blockchain.fund_round_reward_pool(round_num=round_number, amount_eth=amount_eth)
            
            if tx_hash:
                self.logger.info(f"Successfully funded round {round_number} with {amount_eth} ETH")
                return True, tx_hash
            else:
                self.logger.error(f"Transaction failed for funding round {round_number}")
                return False, None
        except Exception as e:
            self.logger.error(f"Error funding reward pool for round {round_number}: {e}")
            traceback.print_exc()
            return False, None
    
    def log_client_rewards(self, round_number, tx_hash=None):
        """
        Log consolidated information about rewards allocated to clients,
        distinguishing between historical rewards and new rewards from current session.
        
        Args:
            round_number: The federated learning round number
            tx_hash: The allocation transaction hash (optional)
        """
        try:
            # Get all client contributions for this round
            round_contributions = self.get_round_contributions(round_number)
            active_clients = set(c['client_address'] for c in round_contributions)
            
            # Get the reward pool info
            pool_info = self.get_reward_pool_info(round_number)
            
            # Log allocation summary
            self.logger.info(f"=== Round {round_number} Reward Allocation Summary ===")
            self.logger.info(f"Total pool: {pool_info['total_eth']} ETH")
            self.logger.info(f"Allocated: {pool_info['allocated_eth']} ETH")
            self.logger.info(f"Remaining: {pool_info['remaining_eth']} ETH")
            self.logger.info(f"Total contributions: {len(round_contributions)}")
            
            # Fetch previous rewards data if exists
            previous_rewards = {}
            try:
                # Check if we have stored previous rewards data
                rewards_file = f"metrics/round_{round_number}_previous_rewards.json"
                if os.path.exists(rewards_file):
                    with open(rewards_file, 'r') as f:
                        previous_rewards = json.load(f)
                    self.logger.info(f"Loaded previous rewards data for {len(previous_rewards)} clients")
            except Exception as e:
                self.logger.warning(f"Could not load previous rewards data: {e}")
            
            # Group by client address to consolidate multiple contributions
            current_rewards = {}
            historical_rewards = {}
            total_score = sum(c['score'] for c in contributions)
            
            for contribution in contributions:
                client_address = contribution['client_address']
                score = contribution['score']
                
                # Calculate this client's reward based on contribution proportion
                client_proportion = score / total_score if total_score > 0 else 0
                client_reward = client_proportion * float(pool_info['allocated_eth'])
                
                # Add to client's total for this session
                if client_address in current_rewards:
                    current_rewards[client_address]['total_score'] += score
                    current_rewards[client_address]['reward'] += client_reward
                    current_rewards[client_address]['contributions'] += 1
                else:
                    current_rewards[client_address] = {
                        'total_score': score,
                        'reward': client_reward,
                        'contributions': 1,
                        'previous_reward': previous_rewards.get(client_address, {}).get('total_reward', 0)
                    }
                
                # Calculate historical rewards (current + previous)
                if client_address in historical_rewards:
                    historical_rewards[client_address]['total_score'] += score
                    historical_rewards[client_address]['total_reward'] += client_reward
                    historical_rewards[client_address]['contributions'] += 1
                else:
                    previous_reward = previous_rewards.get(client_address, {}).get('total_reward', 0)
                    previous_contribs = previous_rewards.get(client_address, {}).get('contributions', 0)
                    
                    historical_rewards[client_address] = {
                        'total_score': score,
                        'total_reward': client_reward + float(previous_reward),
                        'contributions': 1 + int(previous_contribs)
                    }
            
            # Log NEW rewards from current session
            self.logger.info(f"=== NEW Client Rewards (Current Session) ===")
            new_total_eth = 0
            for client_address, data in current_rewards.items():
                # Calculate the new reward (excluding previous rewards)
                new_reward = data['reward']
                new_total_eth += new_reward
                
                self.logger.info(
                    f"Client {client_address}: {new_reward:.6f} ETH from {data['contributions']} "
                    f"contributions with score {data['total_score']} (NEW in this session)"
                )
            self.logger.info(f"TOTAL NEW REWARDS: {new_total_eth:.6f} ETH")
            
            # Log HISTORICAL CUMULATIVE rewards
            self.logger.info(f"=== HISTORICAL CUMULATIVE Client Rewards (All Sessions) ===")
            for client_address, data in historical_rewards.items():
                self.logger.info(
                    f"Client {client_address}: {data['total_reward']:.6f} ETH from {data['contributions']} "
                    f"contributions with total score {data['total_score']} (cumulative)"
                )
            
            # Update clients with newly received rewards
            updated_rewards = {}
            for client_address, data in historical_rewards.items():
                updated_rewards[client_address] = {
                    'total_reward': data['total_reward'],
                    'contributions': data['contributions']
                }
            
            # Store the updated rewards for future reference
            try:
                rewards_file = f"metrics/round_{round_number}_previous_rewards.json"
                os.makedirs(os.path.dirname(rewards_file), exist_ok=True)
                with open(rewards_file, 'w') as f:
                    json.dump(updated_rewards, f, indent=2)
                self.logger.info(f"Updated rewards data saved to {rewards_file}")
            except Exception as e:
                self.logger.error(f"Error saving rewards data: {e}")
            
            # Add explicit logging to identify active vs inactive clients
            self.logger.info(f"Active clients in round {round_number}: {len(active_clients)}")
            for client in active_clients:
                self.logger.info(f"  - {client}")
            
            # Return the current rewards information
            return {
                'current_rewards': current_rewards,
                'historical_rewards': historical_rewards,
                'total_new_rewards': new_total_eth
            }
                
        except Exception as e:
            self.logger.error(f"Error logging client rewards: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
        
    def calculate_round_specific_rewards(self, round_number):
        """
        Calculate rewards for ONLY the active clients in the current round.
        This is critical for matching expected vs actual rewards.
        """
        # Get contributions for this specific round only
        contributions = self.get_round_contributions(round_number)
        if not contributions:
            self.logger.warning(f"No contributions found for round {round_number}")
            return {}
        
        # Get the reward pool info
        pool_info = self.get_reward_pool_info(round_number)
        total_pool = float(pool_info['total_eth'])
        
        # Calculate total score for this round only
        total_score = sum(c['score'] for c in contributions)
        if total_score == 0:
            self.logger.warning(f"Zero total score for round {round_number}, cannot allocate rewards properly")
            return {}
        
        # Group by client and calculate per-client rewards
        client_rewards = {}
        for contribution in contributions:
            client_address = contribution['client_address']
            score = contribution['score']
            
            # Calculate this client's reward based on proportion of total score
            client_proportion = score / total_score
            client_reward = client_proportion * total_pool
            
            # Add to client's rewards for this round
            if client_address in client_rewards:
                client_rewards[client_address]['score'] += score
                client_rewards[client_address]['reward'] += client_reward
                client_rewards[client_address]['contributions'] += 1
            else:
                client_rewards[client_address] = {
                    'score': score,
                    'reward': client_reward,
                    'contributions': 1
                }
        
        return client_rewards
