"""
Configuration functions for Federated Learning server.
These functions provide configuration for clients during fit and evaluate rounds.
"""

from typing import Dict, Callable, Optional
import logging

logger = logging.getLogger("FL-Server-Config")

def fit_config_fn(round_num: int, ga_stacking: bool = True) -> Callable[[int], Dict]:
    """
    Returns a function that creates configuration for the fit method.
    
    Args:
        round_num: Current round number
        ga_stacking: Whether to use GA-Stacking or not
    
    Returns:
        Function that takes client ID and returns configuration dictionary
    """
    def fit_config(client_id: int) -> Dict:
        """
        Create configuration for fit method.
        
        Args:
            client_id: ID of the client
            
        Returns:
            Configuration dictionary
        """
        config = {
            "server_round": round_num,
            "local_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.001,
            "client_id": client_id,
            "ga_stacking": ga_stacking,
            "validation_split": 0.2,
        }
        
        logger.debug(f"Created fit config for client {client_id} in round {round_num}")
        return config
        
    return fit_config


def evaluate_config_fn(round_num: int) -> Callable[[int], Dict]:
    """
    Returns a function that creates configuration for the evaluate method.
    
    Args:
        round_num: Current round number
    
    Returns:
        Function that takes client ID and returns configuration dictionary
    """
    def evaluate_config(client_id: int) -> Dict:
        """
        Create configuration for evaluate method.
        
        Args:
            client_id: ID of the client
            
        Returns:
            Configuration dictionary
        """
        config = {
            "server_round": round_num,
            "client_id": client_id,
        }
        
        logger.debug(f"Created evaluate config for client {client_id} in round {round_num}")
        return config
        
    return evaluate_config