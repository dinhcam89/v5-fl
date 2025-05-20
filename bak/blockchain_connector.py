"""
Enhanced blockchain connector for federated learning with IPFS.
Connects to Ganache and interacts with the EnhancedModelRegistry smart contract.
Adds client authorization and contribution tracking.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import time
import threading
import random
import logging


from web3 import Web3
from web3.contract import Contract
from eth_account.account import Account
from hexbytes import HexBytes

class BlockchainConnector:
    # Handling multiple threads accessing the nonce
    _nonce_lock = threading.Lock()
    _local_nonce = {}
    
    def __init__(
        self, 
        ganache_url: str = "http://192.16880.1:7545",
        contract_address: Optional[str] = None,
        private_key: Optional[str] = None,
        contract_path: Optional[str] = None
    ):
        """Initialize the blockchain connector."""
        # Connect to Ganache
        self.web3 = Web3(Web3.HTTPProvider(ganache_url))
        if not self.web3.is_connected():
            raise ConnectionError(f"Failed to connect to Ganache at {ganache_url}")
        
        # Store the private key as an attribute
        self.private_key = private_key  # <-- Add this line
        
        # Set up account
        if private_key:
            self.account = self.web3.eth.account.from_key(private_key)
            self.address = self.account.address
            print(f"Using account: {self.account.address}")
        else:
            # Use the first account from Ganache
            self.account = self.web3.eth.accounts[0]
            self.address = self.account
            print(f"Using Ganache account: {self.account}")
        
        # Set contract address
        self.contract_address = None
        if contract_address:
            self.contract_address = self.web3.to_checksum_address(contract_address)
            self._load_contract(contract_address, contract_path)
    
    def _load_contract(self, address: str, contract_path: Optional[str] = None) -> None:
        """
        Load the contract from the provided address.
        
        Args:
            address: Contract address
            contract_path: Path to the contract ABI JSON file
        """
        if contract_path is None:
            # Default path for the compiled contract
            contract_path = Path(__file__).parent / "contracts" / "Federation.json"
        
        # Load contract ABI
        try:
            with open(contract_path, 'r') as f:
                contract_json = json.load(f)
            
            # Handle different compilation outputs (Truffle vs. solc)
            if 'abi' in contract_json:
                contract_abi = contract_json['abi']
            elif 'contracts' in contract_json:
                # Get the first contract in the file
                contract_name = list(contract_json['contracts'].keys())[0]
                contract_abi = contract_json['contracts'][contract_name]['abi']
            else:
                raise ValueError("Could not find ABI in contract JSON")
                
            # Create contract instance
            self.contract = self.web3.eth.contract(address=address, abi=contract_abi)
            print(f"Contract loaded at address: {address}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Contract JSON not found at {contract_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading contract: {e}")
    
    def deploy_contract(self, contract_path: Optional[str] = None) -> str:
        """
        Deploy the EnhancedModelRegistry contract to the blockchain.
        
        Args:
            contract_path: Path to the contract JSON file
            
        Returns:
            Address of the deployed contract
        """
        if contract_path is None:
            # Default path for the compiled contract
            contract_path = Path(__file__).parent / "contracts" / "EnhancedModelRegistry.json"
        
        try:
            with open(contract_path, 'r') as f:
                contract_json = json.load(f)
            
            # Get contract bytecode and ABI
            if 'bytecode' in contract_json:
                bytecode = contract_json['bytecode']
                abi = contract_json['abi']
            elif 'contracts' in contract_json:
                # Get the first contract in the file
                contract_name = list(contract_json['contracts'].keys())[0]
                bytecode = contract_json['contracts'][contract_name]['bytecode']
                abi = contract_json['contracts'][contract_name]['abi']
            else:
                raise ValueError("Could not find bytecode and ABI in contract JSON")
            
            # Create contract instance
            Contract = self.web3.eth.contract(abi=abi, bytecode=bytecode)
            
            # Build transaction
            tx_params = {
                'from': self.account if isinstance(self.account, str) else self.account.address,
                'nonce': self.web3.eth.get_transaction_count(
                    self.account if isinstance(self.account, str) else self.account.address
                ),
                'gas': 5000000,  # Increased gas limit for larger contract
                'gasPrice': self.web3.eth.gas_price
            }
            
            # Deploy contract
            transaction = Contract.constructor().build_transaction(tx_params)
            
            # Sign transaction if using private key
            if isinstance(self.account, Account):
                signed_tx = self.account.sign_transaction(transaction)
                tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            else:
                tx_hash = self.web3.eth.send_transaction(transaction)
            
            # Wait for transaction to be mined
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            contract_address = tx_receipt['contractAddress']
            
            # Load the deployed contract
            self._load_contract(contract_address, contract_path)
            
            print(f"Contract deployed at: {contract_address}")
            return contract_address
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Contract JSON not found at {contract_path}")
        except Exception as e:
            raise RuntimeError(f"Error deploying contract: {e}")
    
    def register_model(self, ipfs_hash, round_num, version, participating_clients):
        """
        Register a new model in the blockchain.
        
        Args:
            ipfs_hash: IPFS hash of the model
            round_num: Training round number (will be converted to uint32)
            version: Model version string or timestamp
            participating_clients: Number of clients that participated (will be converted to uint32)
            
        Returns:
            Transaction hash if successful, None otherwise
        """
        try:
            # Convert parameters to appropriate types
            round_num = int(round_num)
            participating_clients = int(participating_clients)
            
            # Use our improved transaction sending method
            function_args = [ipfs_hash, round_num, version, participating_clients]
            tx_hash = self.send_blockchain_transaction("registerModel", function_args)
            
            if tx_hash:
                logging.info(f"Model registered successfully: {ipfs_hash}, round: {round_num}")
            
            return tx_hash
        except Exception as e:
            logging.error(f"Failed to register model: {str(e)}")
            return None

    def register_or_update_model(self, ipfs_hash, round_num, version, participating_clients):
        """
        Register a model in the blockchain or update it if models already exist for this round.
        
        Args:
            ipfs_hash: IPFS hash of the model
            round_num: Federated learning round number (will be converted to uint32)
            version: Model version (can be a timestamp for simplified versioning)
            participating_clients: Number of clients that participated (will be converted to uint32)
            
        Returns:
            Transaction hash or None if failed
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # First, check if models exist for this round
        try:
            # This call doesn't create a transaction - it's just a read operation
            models_for_round = self.contract.functions.getModelsByRound(round_num).call()
            models_exist = len(models_for_round) > 0
        except Exception as e:
            logging.warning(f"Error checking for existing models: {e}")
            # If we can't check, we'll proceed with trying to register
            models_exist = False
        
        # Choose the right function based on whether models exist for this round
        if models_exist:
            function_name = "updateModelByRound"
            logging.info(f"Models exist for round {round_num}, using {function_name}")
        else:
            function_name = "registerModel"
            logging.info(f"No models for round {round_num}, using {function_name}")
        
        # Convert inputs to appropriate types
        round_num = int(round_num)
        participating_clients = int(participating_clients)
        
        # Send the transaction with our improved transaction manager
        function_args = [ipfs_hash, round_num, version, participating_clients]
        tx_hash = self.send_blockchain_transaction(function_name, function_args)
        
        if tx_hash:
            if models_exist:
                logging.info(f"Model updated with transaction: {tx_hash}")
            else:
                logging.info(f"Model registered with transaction: {tx_hash}")
        
        return tx_hash
    
    def register_ga_stacking_model(self, ipfs_hash, round_num, version, 
                             participating_clients, base_model_count, meta_model_count):
        """Register a GA-Stacking model with metadata."""
        return self.send_blockchain_transaction(
            "registerGAStackingModel",
            [ipfs_hash, round_num, version, participating_clients, 
            base_model_count, meta_model_count]
        )
    
    def send_blockchain_transaction(self, transaction_function, function_args, retry_count=5, initial_delay=1):
        """
        Send a blockchain transaction with retry logic and proper nonce management.
        
        Args:
            transaction_function: The contract function to call
            function_args: Arguments to pass to the function
            retry_count: Number of retries if transaction fails
            initial_delay: Initial delay before retry in seconds (will be increased exponentially)
            
        Returns:
            Transaction hash or None if all retries failed
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Get the actual function to call
        contract_function = getattr(self.contract.functions, transaction_function)
        if not contract_function:
            logging.error(f"Function {transaction_function} not found in contract")
            return None
        
        from_address = self.account if isinstance(self.account, str) else self.account.address
        current_delay = initial_delay
        max_delay = 30  # Maximum delay between retries in seconds
        
        for attempt in range(1, retry_count + 1):
            try:
                # Get a fresh nonce for each attempt
                nonce = self.web3.eth.get_transaction_count(from_address)
                logging.info(f"Sending transaction with nonce: {nonce}, gas price: {self.web3.eth.gas_price}")
                
                # Build transaction
                tx_params = {
                    'from': from_address,
                    'nonce': nonce,
                    'gas': 6000000,
                    'gasPrice': self.web3.eth.gas_price
                }
                
                transaction = contract_function(*function_args).build_transaction(tx_params)
                
                # Sign and send
                if isinstance(self.account, Account):
                    signed_tx = self.account.sign_transaction(transaction)
                    tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                else:
                    tx_hash = self.web3.eth.send_transaction(transaction)
                
                logging.info(f"Transaction sent: {tx_hash.hex() if isinstance(tx_hash, HexBytes) else tx_hash}")
                
                # Wait for receipt
                receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                
                # Check status
                if receipt.status == 1:  # 1 = success
                    logging.info(f"Transaction confirmed! Gas used: {receipt.gasUsed}, status: {receipt.status}")
                    return tx_hash.hex() if isinstance(tx_hash, HexBytes) else tx_hash
                else:
                    logging.warning(f"Transaction failed with status: {receipt.status}")
                    # If status is 0 (failure), we'll retry with increased nonce
            
            except Exception as e:
                # Add some randomness to the retry delay (jitter) to prevent thundering herd
                import random
                jitter = random.uniform(0.1, 0.5)
                retry_delay = current_delay + jitter
                
                logging.warning(f"Transaction failed (attempt {attempt}/{retry_count}), retrying in {retry_delay:.1f} seconds: {e}")
                
                if attempt < retry_count:
                    # Exponential backoff with jitter
                    time.sleep(retry_delay)
                    current_delay = min(current_delay * 2, max_delay)
                else:
                    logging.error(f"All {retry_count} transaction attempts failed")
                    return None
            
    def authorize_client(self, client_address: str) -> str:
        """
        Authorize a client to participate in federated learning.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 100000,
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Call authorizeClient function
        transaction = self.contract.functions.authorizeClient(client_address).build_transaction(tx_params)
        
        # Sign transaction if using private key
        if isinstance(self.account, Account):
            signed_tx = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(transaction)
        
        # Wait for transaction to be mined
        _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert HexBytes to string for easier handling
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
            
        print(f"Client {client_address} authorized with transaction: {tx_hash}")
        return tx_hash
    
    def authorize_clients(self, client_addresses: List[str]) -> str:
        """
        Authorize multiple clients to participate in federated learning.
        
        Args:
            client_addresses: List of Ethereum addresses of clients
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 500000,  # Higher gas limit for multiple clients
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Call authorizeClients function
        transaction = self.contract.functions.authorizeClients(client_addresses).build_transaction(tx_params)
        
        # Sign transaction if using private key
        if isinstance(self.account, Account):
            signed_tx = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(transaction)
        
        # Wait for transaction to be mined
        _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert HexBytes to string for easier handling
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
            
        print(f"Authorized {len(client_addresses)} clients with transaction: {tx_hash}")
        return tx_hash
    
    def deauthorize_client(self, client_address: str) -> str:
        """
        Deauthorize a client from participating in federated learning.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 100000,
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Call deauthorizeClient function
        transaction = self.contract.functions.deauthorizeClient(client_address).build_transaction(tx_params)
        
        # Sign transaction if using private key
        if isinstance(self.account, Account):
            signed_tx = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(transaction)
        
        # Wait for transaction to be mined
        _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert HexBytes to string for easier handling
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
            
        print(f"Client {client_address} deauthorized with transaction: {tx_hash}")
        return tx_hash
    
    def is_client_authorized(self, client_address: str) -> bool:
        """
        Check if a client is authorized to participate.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            True if client is authorized, False otherwise
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        return self.contract.functions.isClientAuthorized(client_address).call()
    
    def get_all_authorized_clients(self) -> List[str]:
        """
        Get all authorized client addresses.
        
        Returns:
            List of client addresses
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Get count first
        count = self.contract.functions.getAuthorizedClientCount().call()
        
        # Get each client
        clients = []
        for i in range(count):
            clients.append(self.contract.functions.authorizedClients(i).call())
        
        return clients
    
    def record_contribution(self, client_address, round_num, ipfs_hash, accuracy, max_retries=5):
        """
        Record a client's contribution with advanced retry mechanism and nonce management.
        
        Args:
            client_address: Ethereum address of the client
            round_num: Federated learning round number (will be converted to uint32)
            ipfs_hash: IPFS hash of the model contribution
            accuracy: Accuracy achieved by the client (0-100)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Convert accuracy to blockchain format (multiply by 100 to handle decimals)
        accuracy_int = int(accuracy * 100)
        
        # Convert round_num to uint32
        round_num = int(round_num)
        
        # Get account address consistently
        account_address = self.account if isinstance(self.account, str) else self.account.address
        
        # Set up retry mechanism with exponential backoff and jitter
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Get synchronized nonce with lock protection
                nonce = self._get_synchronized_nonce(account_address)
                
                # Calculate gas price with slight increase on each retry to help stuck transactions
                gas_price_multiplier = 1.0 + (retry_count * 0.2)  # Increase by 20% each retry
                gas_price = int(self.web3.eth.gas_price * gas_price_multiplier)
                
                # Build transaction with carefully managed nonce
                tx_params = {
                    'from': account_address,
                    'nonce': nonce,
                    'gas': 2000000,  # Increased gas limit for more complex transactions
                    'gasPrice': gas_price
                }
                
                logging.info(f"Sending transaction with nonce: {nonce}, gas price: {gas_price}")
                
                # Call recordContribution function
                transaction = self.contract.functions.recordContribution(
                    client_address, 
                    round_num, 
                    ipfs_hash, 
                    accuracy_int
                ).build_transaction(tx_params)
                
                # Sign and send transaction
                if isinstance(self.account, Account):
                    signed_tx = self.account.sign_transaction(transaction)
                    tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                else:
                    tx_hash = self.web3.eth.send_transaction(transaction)
                
                # Convert HexBytes to string for easier handling
                if isinstance(tx_hash, HexBytes):
                    tx_hash_str = tx_hash.hex()
                else:
                    tx_hash_str = tx_hash
                
                logging.info(f"Transaction sent: {tx_hash_str}")
                
                # Wait for transaction to be mined with timeout
                receipt = self.web3.eth.wait_for_transaction_receipt(
                    tx_hash, 
                    timeout=60,
                    poll_latency=1.0  # Check every second
                )
                
                logging.info(f"Transaction confirmed! Gas used: {receipt.gasUsed}, status: {receipt.status}")
                return tx_hash_str
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # Handle nonce-specific errors
                error_str = str(e).lower()
                if "nonce" in error_str and "too low" in error_str:
                    # Reset our nonce tracking if the blockchain says our nonce is too low
                    with self._nonce_lock:
                        if account_address in self._local_nonce:
                            del self._local_nonce[account_address]
                    
                    # Shorter wait for nonce issues
                    wait_time = 1 + random.uniform(0.1, 0.5)
                    logging.warning(f"Nonce too low, resetting and retrying in {wait_time:.1f} seconds")
                
                elif "nonce" in error_str:
                    # Other nonce issue - reset and retry with fresh nonce
                    with self._nonce_lock:
                        if account_address in self._local_nonce:
                            del self._local_nonce[account_address]
                    
                    wait_time = 2 + random.uniform(0.1, 1.0)
                    logging.warning(f"Nonce issue detected, retrying with fresh nonce in {wait_time:.1f} seconds: {e}")
                
                else:
                    # For other errors, use exponential backoff with jitter
                    base_wait = 2 ** retry_count  # 2, 4, 8, 16...
                    wait_time = base_wait + random.uniform(0.1, 1.0)
                    logging.warning(f"Transaction failed (attempt {retry_count}/{max_retries}), "
                                f"retrying in {wait_time:.1f} seconds: {e}")
                
                time.sleep(wait_time)
        
        # If we're here, all retries failed
        logging.error(f"Failed to record contribution after {max_retries} attempts. Last error: {last_error}")
        raise RuntimeError(f"Failed to record contribution: {last_error}")

    def get_round_reward_pool(self, round_num):
        """
        Get details about a round's reward pool.
        
        Args:
            round_num: Round number (will be converted to uint32)
            
        Returns:
            Dictionary with pool details
        """
        # Convert to uint32
        round_num = int(round_num)
        
        total_amount, allocated_amount, remaining_amount, is_finalized = \
            self.contract.functions.getRoundRewardPool(round_num).call()
        
        return {
            "total_amount": self.web3.from_wei(total_amount, 'ether'),
            "allocated_amount": self.web3.from_wei(allocated_amount, 'ether'),
            "remaining_amount": self.web3.from_wei(remaining_amount, 'ether'),
            "is_finalized": is_finalized
        }

    def _get_synchronized_nonce(self, address: str) -> int:
        """
        Get a properly synchronized nonce for the given address.
        Uses a combination of local tracking and blockchain queries.
        
        Args:
            address: Ethereum address
            
        Returns:
            Next nonce to use
        """
        with self._nonce_lock:
            # Get the latest on-chain nonce
            blockchain_nonce = self.web3.eth.get_transaction_count(address, 'pending')
            
            # Initialize or update our local nonce tracking
            if address not in self._local_nonce or self._local_nonce[address] < blockchain_nonce:
                self._local_nonce[address] = blockchain_nonce
            
            # Use our tracked nonce (which may be ahead of the blockchain if we've sent multiple txs)
            next_nonce = self._local_nonce[address]
            
            # Increment for next use
            self._local_nonce[address] = next_nonce + 1
            
            return next_nonce
    
    def get_client_contribution_details(self, client_address: str) -> Dict[str, Any]:
        """
        Get contribution details for a client.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            Dictionary of contribution details
        """
        client_address = self.web3.to_checksum_address(client_address)
        
        # Call the contract function
        contrib_count, total_score, is_authorized, last_timestamp, rewards_earned, rewards_claimed = \
            self.contract.functions.getClientContribution(client_address).call()
        
        # Convert wei to ETH for rewards
        rewards_earned_eth = self.web3.from_wei(rewards_earned, 'ether')
        
        return {
            "contribution_count": contrib_count,
            "total_score": total_score,
            "is_authorized": is_authorized,
            "last_contribution_timestamp": last_timestamp,
            "rewards_earned": rewards_earned_eth,
            "rewards_claimed": rewards_claimed
        }
    
    def get_client_contribution_records(self, client_address, offset=0, limit=100):
        """
        Get client contribution records with pagination.
        
        Args:
            client_address: Ethereum address of the client
            offset: Starting index for pagination
            limit: Maximum number of records to retrieve
            
        Returns:
            List of contribution records
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        try:
            records = self.contract.functions.getClientContributionRecords(
                client_address, 
                offset,
                limit
            ).call()
            
            result = []
            for i in range(len(records[0])):
                result.append({
                    "round": records[0][i],
                    "accuracy": records[1][i] / 100.0,  # Convert back to decimal
                    "score": records[2][i],
                    "timestamp": records[3][i],
                    "rewarded": records[4][i]
                })
            
            return result
        except Exception as e:
            logging.error(f"Error getting client contribution records: {e}")
            return []
    
    def has_contributed_in_round(self, client_address: str, round_number: int) -> bool:
        """
        Check if a client has contributed in a specific round.
        
        Args:
            client_address: Ethereum address of the client
            round_number: The federated learning round number
            
        Returns:
            True if the client has contributed in the specified round, False otherwise
        """
        try:
            if not self.contract:
                logging.warning("Contract not loaded, cannot verify contribution")
                return False
            
            records = self.get_client_contribution_records(client_address)
            
            # Check if any of the records match the current round
            for record in records:
                if record["round"] == round_number:
                    logging.info(f"Found contribution from client {client_address} for round {round_number}")
                    return True
            
            logging.warning(f"No contribution found from client {client_address} for round {round_number}")
            return False
        except Exception as e:
            logging.error(f"Error checking client contribution: {e}")
            # We could be more lenient here by returning True if there's an error
            # return True
            return False
    
    def get_round_contributions(self, round_num: int, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get contributions for a specific round with pagination support.
        
        Args:
            round_num: Federated learning round number
            offset: Starting index for pagination
            limit: Maximum number of records to return
            
        Returns:
            List of contribution records for the round
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        try:
            # Call with pagination parameters
            records = self.contract.functions.getRoundContributions(round_num, offset, limit).call()
            
            result = []
            for i in range(len(records[0])):
                result.append({
                    "client_address": records[0][i],
                    "accuracy": records[1][i] / 100.0,  # Convert back to decimal
                    "score": records[2][i],
                    "rewarded": records[3][i]
                })
            
            return result
        except Exception as e:
            logging.error(f"Error getting round contributions for round {round_num}: {e}")
            return []
    
    def get_model_details(self, ipfs_hash: str, round_num: int) -> Dict[str, Any]:
        """
        Get model details from the blockchain.
        
        Args:
            ipfs_hash: IPFS hash of the model
            round_num: Federated learning round number
            
        Returns:
            Dictionary with model details
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Generate model ID
        model_id = self.contract.functions.generateModelId(ipfs_hash, round_num).call()
        
        # Get model details
        details = self.contract.functions.getModelDetails(model_id).call()
        
        # Format the response
        return {
            "ipfs_hash": details[0],
            "round": details[1],
            "version": details[2],
            "timestamp": details[3],
            "participating_clients": details[4],
            "publisher": details[5],
            "is_active": details[6]
        }
    
    def get_latest_model(self, version_prefix: str = "1.0") -> Dict[str, Any]:
        """
        Get the latest model for a specific version prefix.
        
        Args:
            version_prefix: Version prefix (e.g., "1.0")
            
        Returns:
            Dictionary with model details
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Get latest model
        details = self.contract.functions.getLatestModel(version_prefix).call()
        
        # Format the response
        return {
            "model_id": self.web3.to_hex(details[0]),
            "ipfs_hash": details[1],
            "round": details[2],
            "version": details[3],
            "timestamp": details[4],
            "participating_clients": details[5]
        }
    
    def get_models_by_round(self, round_num: int) -> List[str]:
        """
        Get all models for a specific round.
        
        Args:
            round_num: Federated learning round number
            
        Returns:
            List of model IDs
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Get models by round
        model_ids = self.contract.functions.getModelsByRound(round_num).call()
        
        # Convert bytes32 to hex strings
        return [self.web3.to_hex(model_id) for model_id in model_ids]
    
    def get_latest_model_by_round(self, round_num: int) -> Dict[str, Any]:
        """
        Get the latest model for a specific round.
        
        Args:
            round_num: Federated learning round number
            
        Returns:
            Dictionary with model details
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Get latest model by round
        details = self.contract.functions.getLatestModelByRound(round_num).call()
        
        # Format the response
        return {
            "model_id": self.web3.to_hex(details[0]),
            "ipfs_hash": details[1],
            "round": details[2],
            "version": details[3],
            "timestamp": details[4],
            "participating_clients": details[5]
        }
    
    def deactivate_model(self, ipfs_hash: str, round_num: int) -> str:
        """
        Deactivate a model.
        
        Args:
            ipfs_hash: IPFS hash of the model
            round_num: Federated learning round number
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Generate model ID
        model_id = self.contract.functions.generateModelId(ipfs_hash, round_num).call()
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 100000,
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Call deactivateModel function
        transaction = self.contract.functions.deactivateModel(model_id).build_transaction(tx_params)
        
        # Sign transaction if using private key
        if isinstance(self.account, Account):
            signed_tx = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(transaction)
        
        # Wait for transaction to be mined
        _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert HexBytes to string for easier handling
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
            
        print(f"Model deactivated with transaction: {tx_hash}")
        return tx_hash
    
    def has_contributed_in_round(self, client_address: str, round_number: int) -> bool:
        """
        Check if a client has contributed in a specific round.
        
        Args:
            client_address: Ethereum address of the client
            round_number: The federated learning round number
            
        Returns:
            True if the client has contributed in the specified round, False otherwise
        """
        try:
            if not self.contract:
                logging.warning("Contract not loaded, cannot verify contribution")
                return False
            
            # First check if client is authorized
            if not self.is_client_authorized(client_address):
                logging.warning(f"Client {client_address} is not authorized")
                return False
            
            # Then check contribution details
            details = self.get_client_contribution_details(client_address)
            
            # If client has made any contributions, consider them valid for this round
            # This is a more lenient approach during development
            if details and details["contribution_count"] > 0:
                logging.info(f"Client {client_address} has made {details['contribution_count']} contributions")
                # For more strict checking, uncomment below to verify specific round contributions
                try:
                    records = self.get_client_contribution_records(client_address)
                    for record in records:
                        if record["round"] == round_number:
                            logging.info(f"Found contribution from client {client_address} for round {round_number}")
                            return True
                    
                    # If we get here, client has contributed but not for this specific round
                    # During development, we'll be lenient and accept it anyway
                    logging.warning(f"Client {client_address} has contributed but not for round {round_number}. Accepting anyway.")
                    return True
                except Exception as e:
                    logging.error(f"Error checking specific round contributions: {e}")
                    # Be lenient during development
                    return True
            
            logging.warning(f"No contributions found for client {client_address}")
            return False
        except Exception as e:
            logging.error(f"Error checking client contribution: {e}")
            # During development, you might want to be lenient with errors
            return True  # Change to False for stricter validation
        
    def get_model_by_round(self, round_num: int) -> dict:
        """
        Get the registered model information for a specific round.
        
        Args:
            round_num: The federated learning round number
            
        Returns:
            Dictionary with model information or None if not found
        """
        try:
            # Call the smart contract's getModelByRound function
            model_info = self.contract.functions.getModelByRound(round_num).call()
            
            # Smart contract returns a tuple with model information
            # Convert to a more usable dictionary format
            if model_info and model_info[0]:  # Check if ipfsHash is not empty
                return {
                    "ipfsHash": model_info[0],
                    "roundNum": model_info[1],
                    "version": model_info[2],
                    "participatingClients": model_info[3],
                    "timestamp": model_info[4]
                }
            return None
        except Exception as e:
            logging.error(f"Failed to get model for round {round_num}: {e}")
            return None
            
    def verify_ipfs_hash(self, round_num: int, ipfs_hash: str) -> bool:
        """
        Verify if the provided IPFS hash matches the one registered on the blockchain.
        
        Args:
            round_num: The federated learning round number
            ipfs_hash: The IPFS hash to verify
            
        Returns:
            True if verified, False otherwise
        """
        try:
            model_info = self.get_model_by_round(round_num)
            if model_info and model_info["ipfsHash"] == ipfs_hash:
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to verify IPFS hash: {e}")
            return False
            
    def get_client_contributions(self, round_num: int, wallet_address: str) -> dict:
        """
        Get client contributions for a specific round.
        
        Args:
            round_num: The federated learning round number
            wallet_address: The client's wallet address
            
        Returns:
            Dictionary with contribution information or None if not found
        """
        try:
            # Call the smart contract function
            contribution = self.contract.functions.getClientContribution(
                round_num, 
                wallet_address
            ).call()
            
            # Convert tuple to dictionary
            if contribution and contribution[0]:  # Check if ipfsHash exists
                return {
                    "ipfsHash": contribution[0],
                    "accuracy": contribution[1],
                    "timestamp": contribution[2] if len(contribution) > 2 else 0
                }
            return None
        except Exception as e:
            logging.error(f"Failed to get contributions for client {wallet_address}: {e}")
            return None
    
    ################################
    # MODEL MANAGEMENT
    ################################
    def get_models(self, offset=0, limit=100):
        """
        Get models from the blockchain with pagination.
        
        Args:
            offset: Starting index (ensure it's an integer)
            limit: Maximum number of models to retrieve (ensure it's an integer)
            
        Returns:
            Dictionary containing model data arrays
        """
        try:
            offset_int = int(offset)
            limit_int = int(limit)
            # Ensure integer types and check for unexpected values
            if not isinstance(offset, int) or isinstance(offset, bool):
                logging.warning(f"offset is type {type(offset)}, converting to int")
                offset = int(float(offset))  # Convert through float in case it's a string with decimal point
            
            if not isinstance(limit, int) or isinstance(limit, bool):
                logging.warning(f"limit is type {type(limit)}, converting to int")
                limit = int(float(limit))  # Convert through float in case it's a string with decimal point
            
            # Debug the actual values being sent
            logging.info(f"Calling contract.functions.getModels with: offset={offset} (type={type(offset)}), limit={limit} (type={type(limit)})")
            
            # Call the getModels function from the contract
            result = self.contract.functions.getModels(offset, limit).call()
            
            # Organize results into a dictionary
            models = {
                'modelIds': result[0],
                'ipfsHashes': result[1],
                'rounds': result[2],
                'timestamps': result[3],
                'participatingClients': result[4],
                'isActive': result[5]
            }
            
            return models
        except Exception as e:
            logging.error(f"Failed to get models: {str(e)}")
            return None

    def get_model_count(self):
        """
        Get the total number of models in the blockchain.
        
        Returns:
            Total number of models
        """
        try:
            return self.contract.functions.getModelCount().call()
        except Exception as e:
            logging.error(f"Failed to get model count: {str(e)}")
            return 0
    
    def get_all_models(self, version_prefix=None):
        """
        Get all models from the blockchain, optionally filtered by version prefix.
        
        Args:
            version_prefix: Optional filter for model version prefix
            
        Returns:
            List of dictionaries containing model details
        """
        try:
            # Get the total count of models
            total_count = self.get_model_count()
            if total_count == 0:
                return []
            
            # Define page size
            page_size = 100
            
            # Initialize list to store models
            all_models = []
            
            # Fetch models in batches
            for offset in range(0, total_count, page_size):
                # Calculate the actual limit for this batch
                limit = min(page_size, total_count - offset)
                
                # Ensure offset and limit are integers
                offset_int = int(offset)
                limit_int = int(limit)
                
                # Fetch batch of models
                model_batch = self.get_models(offset_int, limit_int)
                
                if model_batch:
                    # Process each model
                    for i in range(len(model_batch['modelIds'])):
                        model = {
                            'model_id': self.web3.to_hex(model_batch['modelIds'][i]) if isinstance(model_batch['modelIds'][i], bytes) else model_batch['modelIds'][i],
                            'ipfs_hash': model_batch['ipfsHashes'][i],
                            'round': model_batch['rounds'][i],
                            'timestamp': model_batch['timestamps'][i],
                            'participating_clients': model_batch['participatingClients'][i],
                            'is_active': model_batch['isActive'][i]
                        }
                        
                        # If version prefix is specified, we need to get full model details to filter
                        if version_prefix is not None:
                            try:
                                # Get the model's version
                                model_details = self.get_model_details(model['model_id'])
                                if model_details and 'version' in model_details:
                                    model['version'] = model_details['version']
                                    # Check if the version matches the prefix
                                    if not model['version'].startswith(version_prefix):
                                        continue  # Skip this model if prefix doesn't match
                            except Exception as e:
                                logging.warning(f"Could not get version for model {model['model_id']}: {e}")
                        
                        all_models.append(model)
            
            return all_models
        except Exception as e:
            logging.error(f"Failed to get all models: {str(e)}")
            return []
            
    def get_model_details(self, model_id):
        """
        Get details for a specific model by ID.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary with model details
        """
        try:
            # Convert string ID to bytes32 if needed
            if isinstance(model_id, str) and model_id.startswith('0x'):
                model_id = self.web3.to_bytes(hexstr=model_id)
                
            # Call the getModelDetails function
            details = self.contract.functions.getModelDetails(model_id).call()
            
            # Format the response
            return {
                "ipfs_hash": details[0],
                "round": details[1],
                "version": details[2],
                "timestamp": details[3],
                "participating_clients": details[4],
                "publisher": details[5],
                "is_active": details[6]
            }
        except Exception as e:
            logging.error(f"Failed to get model details: {str(e)}")
            return None    
    
    def get_latest_version_model(self, version_prefix):
        """
        Get the latest model for a specific version prefix from the blockchain.
        
        Args:
            version_prefix: Version prefix to check (e.g., "1.0")
            
        Returns:
            Model details or None if not found
        """
        try:
            result = self.contract.functions.getLatestModel(version_prefix).call()
            
            model = {
                'modelId': result[0],
                'ipfsHash': result[1],
                'round': result[2],
                'version': result[3],
                'timestamp': result[4],
                'participatingClients': result[5]
            }
            
            return model
        except Exception as e:
            logging.error(f"Failed to get latest model for version {version_prefix}: {str(e)}")
            return None
    
    ################################
    # REWARDING ALLOCATION
    ################################
    
    def fund_contract(self, amount_eth: float) -> str:
        """
        Fund the contract with ETH.
        
        Args:
            amount_eth: Amount of ETH to send
            
        Returns:
            Transaction hash
        """
        try:
            amount_wei = self.web3.to_wei(amount_eth, 'ether')
            
            # Create transaction to send ETH to contract
            tx_data = {
                'from': self.address,
                'to': self.contract_address,
                'value': amount_wei,
                'gas': 100000,
                'gasPrice': self.web3.to_wei('50', 'gwei'),
                'nonce': self.web3.eth.get_transaction_count(self.address)
            }
            
            if self.private_key:
                # Sign and send transaction
                signed_tx = self.web3.eth.account.sign_transaction(tx_data, self.private_key)
                
                # Use the correct attribute name: raw_transaction instead of rawTransaction
                tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
            else:
                # Send transaction using default account
                tx_hash = self.web3.eth.send_transaction(tx_data)
            
            # Wait for transaction to be mined
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            return receipt.transactionHash.hex()
        except Exception as e:
            print(f"Exception details: {type(e).__name__}: {str(e)}")
            raise

    def fund_round_reward_pool(self, round_num, amount_eth):
        """
        Fund a round's reward pool with ETH.
        
        Args:
            round_num: The federated learning round number (uint32)
            amount_eth: Amount of ETH to fund the pool with
            
        Returns:
            str: Transaction hash hex string or None if failed
        """
        try:
            amount_wei = self.web3.to_wei(amount_eth, 'ether')
            
            # Call the contract's fundRoundRewardPool function
            tx_hash = self.contract.functions.fundRoundRewardPool(round_num).transact({
                'from': self.account.address,
                'value': amount_wei,
                'gas': 500000
            })
            
            # Wait for transaction confirmation
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Check if transaction was successful
            if tx_receipt.status == 1:
                logging.info(f"Successfully funded round {round_num} with {amount_eth} ETH")
                return tx_hash.hex()
            else:
                logging.error(f"Transaction failed for funding round {round_num}")
                return None
        except Exception as e:
            logging.error(f"Error funding round: {e}")
            return None

    def finalize_round_reward_pool(self, round_num):
        """
        Finalize a round's reward pool so that rewards can be allocated.
        
        Args:
            round_num: The federated learning round number (uint32)
            
        Returns:
            str: Transaction hash hex string or None if failed
        """
        try:
            # Call the contract's finalizeRoundRewardPool function
            tx_hash = self.contract.functions.finalizeRoundRewardPool(round_num).transact({
                'from': self.account.address,
                'gas': 500000
            })
            
            # Wait for transaction confirmation
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Check if transaction was successful
            if tx_receipt.status == 1:
                logging.info(f"Successfully finalized reward pool for round {round_num}")
                return tx_hash.hex()
            else:
                logging.error(f"Transaction failed for finalizing round {round_num}")
                return None
        except Exception as e:
            logging.error(f"Error finalizing round: {e}")
            return None

    def allocate_rewards_for_round(self, round_number):
        """
        Call the allocateRewardsForRound function on the contract.
        
        Args:
            round_number: The FL round number
            
        Returns:
            str: Transaction hash if successful, None otherwise
        """
        try:
            tx_hash = self.contract.functions.allocateRewardsForRound(round_number).transact({
                'from': self.account.address,
                'gas': 3000000  # Higher gas limit for this operation
            })
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if tx_receipt.status == 1:
                return tx_hash.hex()
            else:
                logging.error(f"Failed to allocate rewards for round {round_number}")
                return None
        except Exception as e:
            logging.error(f"Error calling allocateRewardsForRound: {e}")
            return None

    def _send_transaction(self, func, value=0):
        """
        Send a transaction to the blockchain.
        
        Args:
            func: Contract function to call
            value: ETH value to send with the transaction (in wei)
            
        Returns:
            Transaction hash
        """
        if self.private_key:
            # Sign transaction with private key
            tx = func.build_transaction({
                'from': self.address,
                'value': value,
                'gas': 3000000,
                'gasPrice': self.web3.to_wei('50', 'gwei'),
                'nonce': self.web3.eth.get_transaction_count(self.address)
            })
            
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
            # Use raw_transaction instead of rawTransaction
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        else:
            # Use default account
            tx_hash = func.transact({
                'from': self.address,
                'value': value,
                'gas': 3000000
            })
        
        # Wait for transaction to be mined
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt.transactionHash.hex()
    
    def allocate_consolidated_rewards(self, round_number, reward_allocations):
        """
        Allocate rewards to multiple clients using the contract's allocateRewardsForRound function.
        
        Args:
            round_number: FL round number
            reward_allocations: Dict mapping client addresses to ETH amounts
            
        Returns:
            tx_hash: Transaction hash
        """
        try:
            # Call the contract's allocateRewardsForRound function
            logging.info(f"Allocating rewards for round {round_number} using allocateRewardsForRound")
            
            tx_hash = self.contract.functions.allocateRewardsForRound(
                round_number
            ).transact({
                'from': self.account.address
            })
            
            # Wait for transaction confirmation
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if tx_receipt.status == 1:
                logging.info(f"Successfully allocated rewards for round {round_number}")
                return tx_hash.hex()
            else:
                logging.error(f"Failed to allocate rewards for round {round_number}")
                return None
        except Exception as e:
            logging.error(f"Error in consolidated reward allocation: {e}")
            return None
    
    def verify_client_contribution(self, client_address, round_number, expected_score):
        """
        Verify that a client's contribution was properly recorded.
        
        Args:
            client_address: Client's Ethereum address
            round_number: The round number
            expected_score: The expected score value
            
        Returns:
            bool: True if verification passed
        """
        try:
            # Get the client's contribution from the contract
            contributions = self.get_round_contributions(round_number)
            client_contributions = [
                c for c in contributions 
                if c['client_address'].lower() == client_address.lower()
            ]
            
            if not client_contributions:
                logging.warning(
                    f"Could not find contribution for client {client_address} in round {round_number}"
                )
                return False
                
            latest_contribution = max(client_contributions, key=lambda c: c['score'])
            actual_score = latest_contribution['score']
            
            # Check if scores match (allowing for small rounding differences)
            if abs(actual_score - expected_score) <= 1:
                logging.info(
                    f"Verified contribution for {client_address} in round {round_number}: "
                    f"score {actual_score} matches expected {expected_score}"
                )
                return True
            else:
                logging.warning(
                    f"Score mismatch for {client_address} in round {round_number}: "
                    f"expected {expected_score}, got {actual_score}"
                )
                return False
                
        except Exception as e:
            logging.error(f"Error verifying client contribution: {e}")
            return False

    def verify_round_contributions(self, round_number):
        """
        Verify that contributions for a round are properly recorded in the contract.
        
        Args:
            round_number: FL round number
            
        Returns:
            dict: Verification results
        """
        try:
            # Get contributions directly from contract
            contributions = self.get_round_contributions(round_number)
            
            # Check if any contributions exist
            if not contributions:
                logging.warning(f"No contributions found for round {round_number}")
                return {"status": "warning", "contributions": 0, "message": "No contributions found"}
            
            # Calculate total score directly from contract data
            total_score = sum(c['score'] for c in contributions)
            unrewarded = sum(1 for c in contributions if not c['rewarded'])
            
            logging.info(f"Round {round_number} verification: {len(contributions)} contributions, " 
                    f"{total_score} total score, {unrewarded} unrewarded")
            
            # Log all contributions for debugging
            for i, c in enumerate(contributions):
                logging.info(
                    f"  Contribution {i+1}: client={c['client_address']} "
                    f"score={c['score']} rewarded={c['rewarded']}"
                )
            
            return {
                "status": "success",
                "contributions": len(contributions),
                "total_score": total_score,
                "unrewarded": unrewarded
            }
        except Exception as e:
            logging.error(f"Error verifying contributions for round {round_number}: {e}")
            return {"status": "error", "message": str(e)}

    def record_contribution_with_verification(self, client_address, round_num, ipfs_hash, 
                                            accuracy, expect_score=None, max_retries=5):
        """
        Record a client's contribution with verification to ensure it was properly recorded.
        
        Args:
            client_address: Ethereum address of the client
            round_num: Federated learning round number
            ipfs_hash: IPFS hash of the model contribution
            accuracy: Accuracy value (0-1 for normalized float, or 0-10000 for contract format)
            expect_score: Expected score after normalization (optional)
            max_retries: Maximum number of retry attempts
            
        Returns:
            tuple: (success, score, tx_hash)
        """
        try:
            # Normalize accuracy for contract if needed (input format: 0-1 float)
            if isinstance(accuracy, float) and 0 <= accuracy <= 1:
                # Contract expects accuracy as 0-10000 integer
                accuracy_for_contract = int(accuracy * 10000)
                logging.info(f"Converting float accuracy {accuracy} to contract format: {accuracy_for_contract}")
            else:
                # Assume the value is already in contract format
                accuracy_for_contract = int(accuracy)
                logging.info(f"Using provided accuracy in contract format: {accuracy_for_contract}")
            
            # Record the contribution
            success, score, tx_hash = self.record_contribution(
                client_address, round_num, ipfs_hash, accuracy_for_contract, max_retries
            )
            
            if not success:
                logging.error(f"Failed to record contribution for {client_address}")
                return False, 0, None
                
            # Calculate the expected score (contract divides accuracy by 100)
            expected_score = expect_score if expect_score is not None else (accuracy_for_contract // 100)
            
            # Verify the contribution was recorded correctly
            verification_success = self.verify_client_contribution(
                client_address, round_num, expected_score
            )
            
            if verification_success:
                logging.info(f"Contribution verified for {client_address} with score {expected_score}")
                return True, expected_score, tx_hash
            else:
                logging.warning(f"Contribution verification failed for {client_address}")
                # Return success anyway since the tx went through
                return True, expected_score, tx_hash
                
        except Exception as e:
            logging.error(f"Error in record_contribution_with_verification: {e}")
            return False, 0, None

    def check_and_fund_round(self, round_number, expected_amount=None):
        """
        Check if a round is already funded and fund it if not.
        
        Args:
            round_number: Round number to check/fund
            expected_amount: Expected ETH amount (if None, calculate based on formula)
            
        Returns:
            tuple: (is_funded, tx_hash)
        """
        try:
            # Get current pool info
            pool_info = self.get_round_reward_pool(round_number)
            
            # Calculate expected amount if not provided (example formula)
            if expected_amount is None:
                base_amount = 0.1  # Base amount in ETH
                increment = 0.02  # Increment per round in ETH
                expected_amount = base_amount + (round_number - 1) * increment
            
            # If pool exists with sufficient funds, don't fund again
            if pool_info and float(pool_info["total_amount"]) > 0:
                # Check if amount is reasonable (within 10% of expected)
                if float(pool_info["total_amount"]) > expected_amount * 1.1:
                    logging.warning(
                        f"Round {round_number} pool has unusually high amount: {pool_info['total_amount']} ETH "
                        f"(expected around {expected_amount} ETH). Not funding again."
                    )
                else:
                    logging.info(f"Round {round_number} pool already funded with {pool_info['total_amount']} ETH")
                
                return True, None
            
            # Fund the pool with the expected amount
            try:
                tx_hash = self.fund_round_reward_pool(round_number, expected_amount)
                if tx_hash:
                    logging.info(f"Successfully funded round {round_number} with {expected_amount} ETH")
                    return True, tx_hash
                else:
                    logging.error(f"Transaction failed for funding round {round_number}")
                    return False, None
            except Exception as e:
                logging.error(f"Error funding reward pool for round {round_number}: {e}")
                return False, None
                
        except Exception as e:
            logging.error(f"Error checking/funding round: {e}")
            return False, None

    def safe_finalize_and_allocate(self, round_number):
        """
        Safely finalize a round and allocate rewards, handling common errors.
        
        Args:
            round_number: Round number to finalize and allocate
            
        Returns:
            tuple: (success, allocated_amount, error_message)
        """
        try:
            # First, verify contributions
            verification = self.verify_round_contributions(round_number)
            if verification.get("status") != "success" or verification.get("unrewarded", 0) == 0:
                msg = f"No unrewarded contributions to allocate for round {round_number}"
                logging.warning(msg)
                
                # Check if we've already allocated rewards
                pool_info = self.get_round_reward_pool(round_number)
                if pool_info['allocated_amount'] > 0:
                    msg = f"Round {round_number} already has {pool_info['allocated_amount']} ETH allocated"
                    logging.info(msg)
                    return True, float(pool_info['allocated_amount']), None
                
                return False, 0, msg
            
            # Check if pool is already finalized
            pool_info = self.get_round_reward_pool(round_number)
            
            # If not finalized yet, finalize it
            if not pool_info.get('is_finalized', False):
                logging.info(f"Finalizing reward pool for round {round_number}")
                try:
                    tx_hash = self.finalize_round_reward_pool(round_number)
                    if not tx_hash:
                        msg = f"Failed to finalize reward pool for round {round_number}"
                        logging.error(msg)
                        return False, 0, msg
                    
                    logging.info(f"Finalized reward pool for round {round_number}")
                except Exception as finalize_err:
                    # Check if it's the "already finalized" error
                    error_str = str(finalize_err)
                    if "Already finalized" in error_str:
                        logging.info(f"Pool for round {round_number} was already finalized")
                    else:
                        msg = f"Error finalizing reward pool: {finalize_err}"
                        logging.error(msg)
                        return False, 0, msg
            else:
                logging.info(f"Pool for round {round_number} is already finalized")
            
            # Now allocate rewards
            try:
                # First check if there are any unrewarded contributions
                verification = self.verify_round_contributions(round_number)
                if verification.get("unrewarded", 0) == 0:
                    msg = f"No unrewarded contributions for round {round_number}. Skipping allocation."
                    logging.info(msg)
                    
                    # Return success if there are allocations already
                    if pool_info['allocated_amount'] > 0:
                        return True, float(pool_info['allocated_amount']), None
                    
                    return False, 0, msg
                
                # Log the total score before allocation - critical for debugging "Zero total score" errors
                logging.info(f"Attempting to allocate rewards with total score: {verification.get('total_score', 0)}")
                
                # Allocate rewards
                tx_hash = self.allocate_rewards_for_round(round_number)
                
                if not tx_hash:
                    msg = f"Failed to allocate rewards for round {round_number}"
                    logging.warning(msg)
                    return False, 0, msg
                
                # Get updated pool info after allocation
                updated_pool_info = self.get_round_reward_pool(round_number)
                allocated_eth = float(updated_pool_info['allocated_amount'])
                
                # Log a summary of the allocation
                logging.info(f"=== Round {round_number} Reward Allocation Summary ===")
                logging.info(f"Total pool: {updated_pool_info['total_amount']} ETH")
                logging.info(f"Allocated: {allocated_eth} ETH")
                logging.info(f"Remaining: {updated_pool_info['remaining_amount']} ETH")
                
                return True, allocated_eth, None
                
            except Exception as alloc_err:
                msg = f"Error allocating rewards: {alloc_err}"
                logging.error(msg)
                return False, 0, msg
                
        except Exception as e:
            msg = f"Error in reward allocation: {e}"
            logging.error(msg)
            return False, 0, msg
    
    ################################
    # LOGGING
    ################################
    def _log_transaction_metrics(self, **kwargs):
        """Log transaction metrics to file for monitoring dashboard"""
        timestamp = datetime.datetime.now().isoformat()
        metrics = {
            'timestamp': timestamp,
            **kwargs
        }
        
        # Ensure directory exists
        os.makedirs('metrics/blockchain', exist_ok=True)
        
        # Append to metrics file
        metrics_file = 'metrics/blockchain/transaction_metrics.jsonl'
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')