#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Client management tool for federated learning with ETH rewards.
"""

import os
import json
import argparse
import sys
import csv
from typing import Dict, List, Optional, Any
from tabulate import tabulate
from datetime import datetime

from blockchain_connector import BlockchainConnector

def load_blockchain_connector(
    ganache_url: str = "http://192.168.80.1:7545",
    contract_address: Optional[str] = None,
    private_key: Optional[str] = None,
) -> BlockchainConnector:
    """
    Load the blockchain connector.
    
    Args:
        ganache_url: Ganache blockchain URL
        contract_address: Address of deployed Federation contract
        private_key: Private key for blockchain transactions
        
    Returns:
        BlockchainConnector instance
    """
    # Check if contract address is provided or stored in file
    if contract_address is None:
        try:
            with open("contract_address.txt", "r") as f:
                contract_address = f.read().strip()
                print(f"Loaded contract address from file: {contract_address}")
        except FileNotFoundError:
            print("Error: No contract address provided or found in file")
            print("Please specify a contract address with --contract-address")
            sys.exit(1)
    
    # Initialize blockchain connector
    try:
        blockchain_connector = BlockchainConnector(
            ganache_url=ganache_url,
            contract_address=contract_address,
            private_key=private_key,
        )
        print(f"Connected to blockchain at {ganache_url}")
        print(f"Contract address: {contract_address}")
        return blockchain_connector
    except Exception as e:
        print(f"Error connecting to blockchain: {e}")
        sys.exit(1)

def list_clients(blockchain_connector: BlockchainConnector, show_details: bool = False) -> None:
    """
    List all authorized clients.
    
    Args:
        blockchain_connector: Blockchain connector instance
        show_details: Whether to show detailed stats for each client
    """
    try:
        # Get all authorized clients
        clients = blockchain_connector.get_all_authorized_clients()
        
        if not clients:
            print("No authorized clients found")
            return
        
        print(f"Found {len(clients)} authorized clients:")
        
        if show_details:
            # Collect detailed stats for each client
            client_details = []
            
            for client in clients:
                try:
                    # Get contribution details
                    details = blockchain_connector.get_client_contribution_details(client)
                    
                    # Add to list
                    client_details.append({
                        "address": client,
                        "contributions": details["contribution_count"],
                        "total_score": details["total_score"],
                        "rewards_earned": details["rewards_earned"],
                        "rewards_claimed": details["rewards_claimed"],
                        "last_contribution": datetime.fromtimestamp(details["last_contribution_timestamp"]).strftime("%Y-%m-%d %H:%M:%S") if details["last_contribution_timestamp"] > 0 else "Never"
                    })
                except Exception as e:
                    print(f"Error getting details for client {client}: {e}")
                    client_details.append({
                        "address": client,
                        "contributions": "Error",
                        "total_score": "Error",
                        "rewards_earned": "Error",
                        "rewards_claimed": "Error",
                        "last_contribution": "Error"
                    })
            
            # Print table
            headers = ["Address", "Contributions", "Total Score", "Rewards Earned (ETH)", "Claimed", "Last Contribution"]
            table_data = [[d["address"], d["contributions"], d["total_score"], d["rewards_earned"], 
                           "Yes" if d["rewards_claimed"] else "No", d["last_contribution"]] for d in client_details]
            
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            # Simple list of addresses
            for i, client in enumerate(clients, 1):
                print(f"{i}. {client}")
    except Exception as e:
        print(f"Error listing clients: {e}")

def authorize_client(blockchain_connector: BlockchainConnector, client_address: str) -> None:
    """
    Authorize a client.
    
    Args:
        blockchain_connector: Blockchain connector instance
        client_address: Ethereum address of the client to authorize
    """
    try:
        # Check if client is already authorized
        if blockchain_connector.is_client_authorized(client_address):
            print(f"Client {client_address} is already authorized")
            return
        
        # Authorize client
        tx_hash = blockchain_connector.authorize_client(client_address)
        print(f"Client {client_address} authorized successfully")
        print(f"Transaction hash: {tx_hash}")
    except Exception as e:
        print(f"Error authorizing client: {e}")

def authorize_clients_from_file(blockchain_connector: BlockchainConnector, filepath: str) -> None:
    """
    Authorize clients from a file.
    
    Args:
        blockchain_connector: Blockchain connector instance
        filepath: Path to the file containing client addresses
    """
    try:
        # Read client addresses from file
        clients = []
        
        # Check file extension
        if filepath.endswith('.csv'):
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].startswith('0x'):
                        clients.append(row[0])
        else:
            # Assume text file with one address per line
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and line.startswith('0x'):
                        clients.append(line)
        
        if not clients:
            print(f"No valid client addresses found in {filepath}")
            return
        
        print(f"Found {len(clients)} client addresses in {filepath}")
        
        # Filter out already authorized clients
        to_authorize = []
        for client in clients:
            if not blockchain_connector.is_client_authorized(client):
                to_authorize.append(client)
        
        if not to_authorize:
            print("All clients are already authorized")
            return
        
        print(f"Authorizing {len(to_authorize)} clients...")
        
        # Authorize clients in batches of 10
        batch_size = 10
        for i in range(0, len(to_authorize), batch_size):
            batch = to_authorize[i:i+batch_size]
            tx_hash = blockchain_connector.authorize_clients(batch)
            print(f"Authorized batch {i//batch_size + 1}/{(len(to_authorize)+batch_size-1)//batch_size}, tx: {tx_hash}")
        
        print(f"Successfully authorized {len(to_authorize)} clients")
    except Exception as e:
        print(f"Error authorizing clients from file: {e}")

def deauthorize_client(blockchain_connector: BlockchainConnector, client_address: str) -> None:
    """
    Deauthorize a client.
    
    Args:
        blockchain_connector: Blockchain connector instance
        client_address: Ethereum address of the client to deauthorize
    """
    try:
        # Check if client is authorized
        if not blockchain_connector.is_client_authorized(client_address):
            print(f"Client {client_address} is not authorized")
            return
        
        # Confirm deauthorization
        confirm = input(f"Are you sure you want to deauthorize client {client_address}? [y/N]: ")
        if confirm.lower() != 'y':
            print("Deauthorization cancelled")
            return
        
        # Deauthorize client
        tx_hash = blockchain_connector.deauthorize_client(client_address)
        print(f"Client {client_address} deauthorized successfully")
        print(f"Transaction hash: {tx_hash}")
    except Exception as e:
        print(f"Error deauthorizing client: {e}")

def show_client_contributions(blockchain_connector: BlockchainConnector, client_address: str) -> None:
    """
    Show detailed contributions for a client.
    
    Args:
        blockchain_connector: Blockchain connector instance
        client_address: Ethereum address of the client
    """
    try:
        # Check if client is authorized
        if not blockchain_connector.is_client_authorized(client_address):
            print(f"Client {client_address} is not authorized")
            return
        
        # Get client contribution details
        details = blockchain_connector.get_client_contribution_details(client_address)
        
        print(f"Client {client_address} contribution summary:")
        print(f"  Total contributions: {details['contribution_count']}")
        print(f"  Total score: {details['total_score']}")
        print(f"  Rewards earned: {details['rewards_earned']} ETH")
        print(f"  Rewards claimed: {'Yes' if details['rewards_claimed'] else 'No'}")
        
        if details["last_contribution_timestamp"] > 0:
            last_contrib = datetime.fromtimestamp(details["last_contribution_timestamp"])
            print(f"  Last contribution: {last_contrib.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("  Last contribution: Never")
        
        # Get contribution records
        records = blockchain_connector.get_client_contribution_records(client_address)
        
        if not records:
            print("No contribution records found")
            return
        
        print(f"\nContribution history ({len(records)} records):")
        
        # Prepare table data
        headers = ["Round", "Accuracy", "Score", "Timestamp", "Rewarded"]
        table_data = []
        
        for record in records:
            timestamp = datetime.fromtimestamp(record["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            table_data.append([
                record["round"],
                f"{record['accuracy']:.2f}%",
                record["score"],
                timestamp,
                "Yes" if record["rewarded"] else "No"
            ])
        
        # Sort by round
        table_data.sort(key=lambda x: x[0])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    except Exception as e:
        print(f"Error showing client contributions: {e}")

def show_round_contributions(blockchain_connector: BlockchainConnector, round_num: int) -> None:
    """
    Show all contributions for a specific round.
    
    Args:
        blockchain_connector: Blockchain connector instance
        round_num: Round number
    """
    try:
        # Get round contributions
        contributions = blockchain_connector.get_round_contributions(round_num)
        
        if not contributions:
            print(f"No contributions found for round {round_num}")
            return
        
        print(f"Found {len(contributions)} contributions for round {round_num}:")
        
        # Prepare table data
        headers = ["Client Address", "Accuracy", "Score", "Rewarded"]
        table_data = []
        
        for contrib in contributions:
            table_data.append([
                contrib["client_address"],
                f"{contrib['accuracy']:.2f}%",
                contrib["score"],
                "Yes" if contrib["rewarded"] else "No"
            ])
        
        # Sort by score (descending)
        table_data.sort(key=lambda x: x[2], reverse=True)
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Show reward pool information if available
        try:
            pool_info = blockchain_connector.get_round_reward_pool(round_num)
            print("\nReward Pool Information:")
            print(f"  Total Amount: {pool_info['total_amount']} ETH")
            print(f"  Allocated Amount: {pool_info['allocated_amount']} ETH")
            print(f"  Remaining Amount: {pool_info['remaining_amount']} ETH")
            print(f"  Finalized: {'Yes' if pool_info['is_finalized'] else 'No'}")
        except Exception as e:
            print(f"\nError getting reward pool information: {e}")
            
    except Exception as e:
        print(f"Error showing round contributions: {e}")

def fund_contract(blockchain_connector: BlockchainConnector, amount_eth: float) -> None:
    """
    Fund the contract with ETH.
    
    Args:
        blockchain_connector: Blockchain connector instance
        amount_eth: Amount of ETH to send
    """
    try:
        # Confirm funding
        confirm = input(f"Are you sure you want to fund the contract with {amount_eth} ETH? [y/N]: ")
        if confirm.lower() != 'y':
            print("Funding cancelled")
            return
        
        # Fund contract
        tx_hash = blockchain_connector.fund_contract(amount_eth)
        print(f"Contract funded with {amount_eth} ETH")
        print(f"Transaction hash: {tx_hash}")
    except Exception as e:
        print(f"Error funding contract: {e}")

def fund_round_reward_pool(blockchain_connector: BlockchainConnector, round_num: int, amount_eth: float) -> None:
    """
    Fund a specific round's reward pool.
    
    Args:
        blockchain_connector: Blockchain connector instance
        round_num: Round number
        amount_eth: Amount of ETH to allocate
    """
    try:
        # Confirm funding
        confirm = input(f"Are you sure you want to fund round {round_num} with {amount_eth} ETH? [y/N]: ")
        if confirm.lower() != 'y':
            print("Funding cancelled")
            return
        
        # Fund round
        tx_hash = blockchain_connector.fund_round_reward_pool(round_num, amount_eth)
        print(f"Round {round_num} funded with {amount_eth} ETH")
        print(f"Transaction hash: {tx_hash}")
    except Exception as e:
        print(f"Error funding round: {e}")

def finalize_round_reward_pool(blockchain_connector: BlockchainConnector, round_num: int) -> None:
    """
    Finalize a round's reward pool.
    
    Args:
        blockchain_connector: Blockchain connector instance
        round_num: Round number
    """
    try:
        # Get reward pool information
        pool_info = blockchain_connector.get_round_reward_pool(round_num)
        
        if pool_info["is_finalized"]:
            print(f"Reward pool for round {round_num} is already finalized")
            return
        
        # Confirm finalization
        confirm = input(f"Are you sure you want to finalize the reward pool for round {round_num}? "
                        f"(Current total: {pool_info['total_amount']} ETH) [y/N]: ")
        if confirm.lower() != 'y':
            print("Finalization cancelled")
            return
        
        # Finalize round
        tx_hash = blockchain_connector.finalize_round_reward_pool(round_num)
        print(f"Reward pool for round {round_num} finalized")
        print(f"Transaction hash: {tx_hash}")
    except Exception as e:
        print(f"Error finalizing round reward pool: {e}")

def allocate_rewards(blockchain_connector: BlockchainConnector, round_num: int) -> None:
    """
    Allocate ETH rewards for a specific round.
    
    Args:
        blockchain_connector: Blockchain connector instance
        round_num: Round number
    """
    try:
        # Check if there are contributions for this round
        contributions = blockchain_connector.get_round_contributions(round_num)
        
        if not contributions:
            print(f"No contributions found for round {round_num}")
            return
        
        # Get reward pool information
        pool_info = blockchain_connector.get_round_reward_pool(round_num)
        
        if not pool_info["is_finalized"]:
            print(f"Reward pool for round {round_num} is not finalized yet")
            print("Please finalize the reward pool first with the 'finalize-pool' command")
            return
        
        # Check if rewards are already allocated
        if all(contrib["rewarded"] for contrib in contributions):
            print(f"Rewards for round {round_num} are already allocated")
            return
        
        # Show reward pool and contributions
        print(f"Round {round_num} Reward Pool: {pool_info['total_amount']} ETH")
        print(f"Number of contributions: {len(contributions)}")
        
        # Calculate estimated reward distribution
        total_score = sum(contrib["score"] for contrib in contributions if not contrib["rewarded"])
        
        if total_score == 0:
            print("No unrewarded contributions with non-zero scores found")
            return
        
        print("\nEstimated reward distribution:")
        headers = ["Client Address", "Score", "Percentage", "Estimated ETH"]
        table_data = []
        
        for contrib in contributions:
            if not contrib["rewarded"] and contrib["score"] > 0:
                percentage = (contrib["score"] / total_score) * 100
                estimated_eth = (contrib["score"] / total_score) * float(pool_info['remaining_amount'])
                table_data.append([
                    contrib["client_address"],
                    contrib["score"],
                    f"{percentage:.2f}%",
                    f"{estimated_eth:.6f}"
                ])
        
        # Sort by score (descending)
        table_data.sort(key=lambda x: x[1], reverse=True)
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Confirm allocation
        confirm = input(f"Are you sure you want to allocate rewards for round {round_num}? [y/N]: ")
        if confirm.lower() != 'y':
            print("Reward allocation cancelled")
            return
        
        # Allocate rewards
        tx_hash = blockchain_connector.allocate_rewards_for_round(round_num)
        print(f"Rewards allocated successfully for round {round_num}")
        print(f"Transaction hash: {tx_hash}")
    except Exception as e:
        print(f"Error allocating rewards: {e}")

def record_mock_contribution(
    blockchain_connector: BlockchainConnector, 
    client_address: str, 
    round_num: int, 
    accuracy: float
) -> None:
    """
    Record a mock contribution for testing.
    
    Args:
        blockchain_connector: Blockchain connector instance
        client_address: Client address
        round_num: Round number
        accuracy: Contribution accuracy (percentage)
    """
    try:
        # Check if client is authorized
        if not blockchain_connector.is_client_authorized(client_address):
            print(f"Client {client_address} is not authorized")
            return
        
        # Generate mock IPFS hash
        import uuid
        mock_ipfs_hash = f"Qm{uuid.uuid4().hex[:34]}"
        
        # Record contribution
        tx_hash, score = blockchain_connector.record_contribution(
            client_address=client_address,
            round_num=round_num,
            ipfs_hash=mock_ipfs_hash,
            accuracy=accuracy
        )
        
        print(f"Recorded mock contribution for client {client_address}")
        print(f"  Round: {round_num}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Score: {score}")
        print(f"  IPFS hash: {mock_ipfs_hash}")
        print(f"  Transaction hash: {tx_hash}")
    except Exception as e:
        print(f"Error recording mock contribution: {e}")

def register_mock_model(
    blockchain_connector: BlockchainConnector,
    round_num: int,
    version: str,
    participating_clients: int
) -> None:
    """
    Register a mock model for testing.
    
    Args:
        blockchain_connector: Blockchain connector instance
        round_num: Round number
        version: Version string
        participating_clients: Number of participating clients
    """
    try:
        # Generate mock IPFS hash
        import uuid
        mock_ipfs_hash = f"Qm{uuid.uuid4().hex[:34]}"
        
        # Register model
        tx_hash = blockchain_connector.register_model(
            ipfs_hash=mock_ipfs_hash,
            round_num=round_num,
            version=version,
            participating_clients=participating_clients
        )
        
        print(f"Registered mock model for round {round_num}")
        print(f"  Version: {version}")
        print(f"  Participating clients: {participating_clients}")
        print(f"  IPFS hash: {mock_ipfs_hash}")
        print(f"  Transaction hash: {tx_hash}")
    except Exception as e:
        print(f"Error registering mock model: {e}")

def export_client_stats(blockchain_connector: BlockchainConnector, filepath: str) -> None:
    """
    Export client statistics to a file.
    
    Args:
        blockchain_connector: Blockchain connector instance
        filepath: Path to export the statistics
    """
    try:
        # Get all authorized clients
        clients = blockchain_connector.get_all_authorized_clients()
        
        if not clients:
            print("No authorized clients found")
            return
        
        # Collect stats for each client
        client_stats = {}
        
        for client in clients:
            try:
                # Get contribution details
                details = blockchain_connector.get_client_contribution_details(client)
                
                # Get contribution records
                records = blockchain_connector.get_client_contribution_records(client)
                
                # Store in stats
                client_stats[client] = {
                    "details": details,
                    "records": records
                }
            except Exception as e:
                print(f"Error getting stats for client {client}: {e}")
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(client_stats, f, indent=2)
            
        print(f"Exported client stats for {len(client_stats)} clients to {filepath}")
    except Exception as e:
        print(f"Error exporting client stats: {e}")

def main():
    parser = argparse.ArgumentParser(description="Client management tool for federated learning with ETH rewards")
    
    # Blockchain connection
    parser.add_argument("--ganache-url", type=str, default="http://192.168.80.1:7545", help="Ganache blockchain URL")
    parser.add_argument("--contract-address", type=str, help="Federation contract address")
    parser.add_argument("--private-key", type=str, help="Private key for blockchain transactions")
    
    # Actions
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # List clients
    list_parser = subparsers.add_parser("list", help="List all authorized clients")
    list_parser.add_argument("--details", action="store_true", help="Show detailed stats for each client")
    
    # Authorize client
    auth_parser = subparsers.add_parser("authorize", help="Authorize a client")
    auth_parser.add_argument("client_address", type=str, help="Ethereum address of the client to authorize")
    
    # Authorize clients from file
    auth_file_parser = subparsers.add_parser("authorize-file", help="Authorize clients from a file")
    auth_file_parser.add_argument("filepath", type=str, help="Path to the file containing client addresses")
    
    # Deauthorize client
    deauth_parser = subparsers.add_parser("deauthorize", help="Deauthorize a client")
    deauth_parser.add_argument("client_address", type=str, help="Ethereum address of the client to deauthorize")
    
    # Show client contributions
    client_contrib_parser = subparsers.add_parser("client-contributions", help="Show detailed contributions for a client")
    client_contrib_parser.add_argument("client_address", type=str, help="Ethereum address of the client")
    
    # Show round contributions
    round_contrib_parser = subparsers.add_parser("round-contributions", help="Show all contributions for a specific round")
    round_contrib_parser.add_argument("round_num", type=int, help="Round number")
    
    # Fund contract
    fund_contract_parser = subparsers.add_parser("fund-contract", help="Fund the contract with ETH")
    fund_contract_parser.add_argument("amount_eth", type=float, help="Amount of ETH to send")
    
    # Fund round reward pool
    fund_round_parser = subparsers.add_parser("fund-round", help="Fund a specific round's reward pool")
    fund_round_parser.add_argument("round_num", type=int, help="Round number")
    fund_round_parser.add_argument("amount_eth", type=float, help="Amount of ETH to allocate")
    
    # Finalize round reward pool
    finalize_pool_parser = subparsers.add_parser("finalize-pool", help="Finalize a round's reward pool")
    finalize_pool_parser.add_argument("round_num", type=int, help="Round number")
    
    # Allocate rewards
    reward_parser = subparsers.add_parser("allocate-rewards", help="Allocate rewards for a specific round")
    reward_parser.add_argument("round_num", type=int, help="Round number")
    
    # Record mock contribution
    mock_contrib_parser = subparsers.add_parser("record-mock", help="Record a mock contribution for testing")
    mock_contrib_parser.add_argument("client_address", type=str, help="Ethereum address of the client")
    mock_contrib_parser.add_argument("round_num", type=int, help="Round number")
    mock_contrib_parser.add_argument("accuracy", type=float, help="Contribution accuracy (percentage)")
    
    # Register mock model
    mock_model_parser = subparsers.add_parser("register-model", help="Register a mock model for testing")
    mock_model_parser.add_argument("round_num", type=int, help="Round number")
    mock_model_parser.add_argument("version", type=str, help="Version string (e.g., '1.0.3')")
    mock_model_parser.add_argument("participating_clients", type=int, help="Number of participating clients")
    
    # Export client stats
    export_parser = subparsers.add_parser("export", help="Export client statistics to a file")
    export_parser.add_argument("filepath", type=str, help="Path to export the statistics")
    
    args = parser.parse_args()
    
    if not args.action:
        parser.print_help()
        return
    
    # Load blockchain connector
    blockchain_connector = load_blockchain_connector(
        ganache_url=args.ganache_url,
        contract_address=args.contract_address,
        private_key=args.private_key,
    )
    
    # Perform the selected action
    if args.action == "list":
        list_clients(blockchain_connector, args.details)
    elif args.action == "authorize":
        authorize_client(blockchain_connector, args.client_address)
    elif args.action == "authorize-file":
        authorize_clients_from_file(blockchain_connector, args.filepath)
    elif args.action == "deauthorize":
        deauthorize_client(blockchain_connector, args.client_address)
    elif args.action == "client-contributions":
        show_client_contributions(blockchain_connector, args.client_address)
    elif args.action == "round-contributions":
        show_round_contributions(blockchain_connector, args.round_num)
    elif args.action == "fund-contract":
        fund_contract(blockchain_connector, args.amount_eth)
    elif args.action == "fund-round":
        fund_round_reward_pool(blockchain_connector, args.round_num, args.amount_eth)
    elif args.action == "finalize-pool":
        finalize_round_reward_pool(blockchain_connector, args.round_num)
    elif args.action == "allocate-rewards":
        allocate_rewards(blockchain_connector, args.round_num)
    elif args.action == "record-mock":
        record_mock_contribution(blockchain_connector, args.client_address, args.round_num, args.accuracy)
    elif args.action == "register-model":
        register_mock_model(blockchain_connector, args.round_num, args.version, args.participating_clients)
    elif args.action == "export":
        export_client_stats(blockchain_connector, args.filepath)

if __name__ == "__main__":
    main()