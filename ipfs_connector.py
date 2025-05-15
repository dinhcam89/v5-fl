import io
import json
import logging
import requests
import pickle
import base64
from typing import Dict, Any, Union, BinaryIO, Optional

logger = logging.getLogger("federated.ipfs")

class IPFSConnector:
    """
    Enhanced IPFS connector for federated learning - optimized for model exchange
    Works with scikit-learn models and generic Python objects
    """
    
    def __init__(self, ipfs_api_url: str = "http://127.0.0.1:5001/api/v0"):
        """
        Initialize the IPFS connector
        
        Args:
            ipfs_api_url: URL to the IPFS API
        """
        self.ipfs_api_url = ipfs_api_url
        
        # Check connection
        try:
            response = requests.post(f"{self.ipfs_api_url}/version")
            response.raise_for_status()
            version = response.json()
            logger.info(f"Connected to IPFS node version: {version.get('Version', 'Unknown')}")
        except requests.RequestException as e:
            logger.error(f"Failed to connect to IPFS node: {e}")
            raise ConnectionError(f"Failed to connect to IPFS node at {ipfs_api_url}")
    
    def add_file(self, file_data: Union[bytes, BinaryIO], filename: Optional[str] = None) -> str:
        """
        Add a file to IPFS
        
        Args:
            file_data: File content as bytes or file-like object
            filename: Optional filename for the file
            
        Returns:
            IPFS hash (CID) of the added file
        """
        files = {}
        
        if isinstance(file_data, bytes):
            if filename:
                files = {'file': (filename, io.BytesIO(file_data))}
            else:
                files = {'file': io.BytesIO(file_data)}
        else:
            if filename:
                files = {'file': (filename, file_data)}
            else:
                files = {'file': file_data}
        
        response = requests.post(
            f"{self.ipfs_api_url}/add",
            files=files
        )
        
        response.raise_for_status()
        result = response.json()
        
        ipfs_hash = result.get('Hash')
        if not ipfs_hash:
            raise ValueError("Failed to get IPFS hash from response")
        
        logger.info(f"Added file to IPFS with hash: {ipfs_hash}")
        return ipfs_hash
    
    def get_file(self, ipfs_hash: str) -> bytes:
        """
        Get a file from IPFS by its hash
        
        Args:
            ipfs_hash: IPFS hash (CID) of the file
            
        Returns:
            File content as bytes
        """
        response = requests.post(
            f"{self.ipfs_api_url}/cat",
            params={'arg': ipfs_hash}
        )
        
        response.raise_for_status()
        
        logger.info(f"Retrieved file from IPFS with hash: {ipfs_hash}")
        return response.content
    
    def pin_file(self, ipfs_hash: str) -> bool:
        """
        Pin a file in IPFS to prevent garbage collection
        
        Args:
            ipfs_hash: IPFS hash (CID) of the file
            
        Returns:
            True if pinning was successful
        """
        response = requests.post(
            f"{self.ipfs_api_url}/pin/add",
            params={'arg': ipfs_hash}
        )
        
        response.raise_for_status()
        result = response.json()
        
        pinned = ipfs_hash in result.get('Pins', [])
        
        if pinned:
            logger.info(f"Pinned file with hash: {ipfs_hash}")
        else:
            logger.warning(f"Failed to pin file with hash: {ipfs_hash}")
        
        return pinned
    
    def upload_model(self, model: Any, model_info: Dict[str, Any] = None) -> str:
        """
        Upload a model to IPFS (works with scikit-learn models)
        
        Args:
            model: The model to upload (e.g., scikit-learn model)
            model_info: Additional info to store with the model
            
        Returns:
            IPFS hash (CID) of the uploaded model
        """
        # Two options for serialization:
        # 1. JSON for metadata + base64 encoded pickle for the model
        # 2. Full JSON serialization of parameters for compatible models
        
        try:
            # Serialize the model to pickle
            model_pickle = pickle.dumps(model)
            model_b64 = base64.b64encode(model_pickle).decode('utf-8')
            
            # Create a dictionary with both model data and metadata
            model_data = {
                "model_format": "pickle_base64",
                "model_data": model_b64,
                "info": model_info or {}
            }
            
            # Convert to JSON
            model_json = json.dumps(model_data)
            
            # Add the model to IPFS
            ipfs_hash = self.add_file(model_json.encode('utf-8'), "model.json")
            
            # Pin the file to prevent garbage collection
            self.pin_file(ipfs_hash)
            
            logger.info(f"Uploaded model to IPFS with hash: {ipfs_hash}")
            
            return ipfs_hash
            
        except Exception as e:
            logger.error(f"Error uploading model to IPFS: {e}")
            raise
    
    def download_model(self, ipfs_hash: str) -> Any:
        """
        Download a model from IPFS
        
        Args:
            ipfs_hash: IPFS hash (CID) of the model
            
        Returns:
            The deserialized model
        """
        try:
            # Get the model data from IPFS
            model_data = self.get_file(ipfs_hash)
            
            # Parse the JSON data
            model_json = json.loads(model_data.decode('utf-8'))
            
            # Check the format
            model_format = model_json.get("model_format", "unknown")
            
            if model_format == "pickle_base64":
                # Decode and unpickle the model
                model_b64 = model_json["model_data"]
                model_pickle = base64.b64decode(model_b64)
                model = pickle.loads(model_pickle)
                logger.info(f"Successfully downloaded and unpickled model from IPFS: {ipfs_hash}")
                return model
            else:
                logger.error(f"Unsupported model format: {model_format}")
                raise ValueError(f"Unsupported model format: {model_format}")
                
        except Exception as e:
            logger.error(f"Error downloading model from IPFS: {e}")
            raise
    
    def get_model_info(self, ipfs_hash: str) -> Dict[str, Any]:
        """
        Get model info from IPFS without downloading the entire model
        
        Args:
            ipfs_hash: IPFS hash (CID) of the model
            
        Returns:
            Dictionary containing the model info
        """
        try:
            # Get the model data from IPFS
            model_data = self.get_file(ipfs_hash)
            
            # Parse the JSON data
            model_json = json.loads(model_data.decode('utf-8'))
            
            # Extract just the info part
            info = model_json.get("info", {})
            
            return info
        except Exception as e:
            logger.error(f"Error getting model info from IPFS: {e}")
            raise
    
    def get_gateway_url(self, ipfs_hash: str) -> str:
        """
        Get a HTTP gateway URL for an IPFS hash
        
        Args:
            ipfs_hash: IPFS hash (CID)
            
        Returns:
            HTTP URL to access the file
        """
        # Local gateway
        return f"http://localhost:8080/ipfs/{ipfs_hash}"
    
    def publish_model_hash(self, ipfs_hash: str, key_name: str = "federated-model") -> str:
        """
        Publish model hash to IPNS for persistent addressing
        
        Args:
            ipfs_hash: IPFS hash (CID) of the model
            key_name: IPNS key name
            
        Returns:
            IPNS address
        """
        try:
            # Check if key exists, create if not
            response = requests.post(
                f"{self.ipfs_api_url}/key/list"
            )
            response.raise_for_status()
            keys = response.json().get("Keys", [])
            
            key_exists = False
            for key in keys:
                if key.get("Name") == key_name:
                    key_exists = True
                    break
            
            if not key_exists:
                # Create a new key
                response = requests.post(
                    f"{self.ipfs_api_url}/key/gen",
                    params={'arg': key_name}
                )
                response.raise_for_status()
                
            # Publish to IPNS
            response = requests.post(
                f"{self.ipfs_api_url}/name/publish",
                params={
                    'arg': ipfs_hash,
                    'key': key_name,
                    'lifetime': '24h'  # Set lifetime of the record
                }
            )
            response.raise_for_status()
            result = response.json()
            
            ipns_name = result.get('Name')
            if not ipns_name:
                raise ValueError("Failed to get IPNS name from response")
            
            logger.info(f"Published model hash {ipfs_hash} to IPNS with name: {ipns_name}")
            return ipns_name
        except Exception as e:
            logger.error(f"Error publishing to IPNS: {e}")
            raise
        
    def verify_hash_exists(self, ipfs_hash):
        """
        Verify that a hash exists in IPFS.
        
        Args:
            ipfs_hash: The IPFS hash to check
            
        Returns:
            bool: True if the hash exists, False otherwise
        """
        try:
            # Use the /files/stat endpoint which works with UnixFS objects
            url = f"{self.ipfs_api_url}/files/stat"
            response = requests.post(url, params={"arg": f"/ipfs/{ipfs_hash}"})
            
            if response.status_code == 200:
                stats = response.json()
                size_bytes = stats.get("Size", 0)
                size_kb = size_bytes / 1024
                logger.info(f"Verified: Hash {ipfs_hash} exists in IPFS")
                logger.info(f"Model size: {size_kb:.2f} KB")
                return True
            else:
                # Try another approach - cat API to see if we can retrieve the content
                cat_url = f"{self.ipfs_api_url}/cat"
                cat_response = requests.post(cat_url, params={"arg": ipfs_hash}, stream=True)
                
                if cat_response.status_code == 200:
                    # We were able to retrieve it, so it exists
                    # Close the connection without downloading the whole file
                    cat_response.close()
                    logger.info(f"Verified: Hash {ipfs_hash} exists in IPFS (via cat API)")
                    return True
                
                logger.warning(f"Hash {ipfs_hash} does not exist in IPFS or is not accessible")
                return False
        except Exception as e:
            logger.error(f"Error verifying hash {ipfs_hash}: {e}")
            return False
        
    def inspect_model_content(self, ipfs_hash):
        """
        Retrieve and inspect the contents of a model stored in IPFS.
        
        Args:
            ipfs_hash: The IPFS hash to inspect
            
        Returns:
            dict: The model information if successfully decoded, None otherwise
        """
        try:
            # Retrieve the content from IPFS
            response = requests.post(
                f"{self.ipfs_api_url}/cat",
                params={"arg": ipfs_hash}
            )
            
            if response.status_code != 200:
                print(f"Failed to retrieve content for hash {ipfs_hash}: {response.text}")
                return None
            
            # Get the content
            content = response.content
            print(f"Retrieved {len(content)} bytes from IPFS hash {ipfs_hash}")
            
            # First try to interpret as JSON
            try:
                import json
                model_json = json.loads(content)
                print(f"Successfully decoded as JSON with keys: {list(model_json.keys())}")
                
                # Check if it's our format with pickled model
                if "model_format" in model_json and model_json["model_format"] == "pickle_base64":
                    print("Contains base64 encoded pickled model")
                    
                    # Don't decode the full model, just report its presence
                    info = model_json.get("info", {})
                    print(f"Model info: {json.dumps(info, indent=2)}")
                    
                    return model_json
                else:
                    print(f"JSON content: {json.dumps(model_json, indent=2)[:1000]}")
                    return model_json
                    
            except json.JSONDecodeError as je:
                print(f"Not valid JSON: {je}")
            
            # Try to load as a pickle object
            try:
                import pickle
                model_data = pickle.loads(content)
                
                # Print model structure
                print(f"Model data type: {type(model_data)}")
                
                if isinstance(model_data, dict):
                    # This is likely a dict with model info
                    keys = list(model_data.keys())
                    print(f"Model contains {len(keys)} keys: {keys}")
                    return model_data
                
                else:
                    print(f"Unpickled object of type: {type(model_data)}")
                    return model_data
                    
            except Exception as e:
                print(f"Failed to load as pickled model: {e}")
                    
            # Try to interpret as plain text
            try:
                text_content = content.decode('utf-8')
                print(f"Content as text (first 500 chars):\n{text_content[:500]}")
                
                # Save full content to a file
                with open(f"ipfs_content_{ipfs_hash}.txt", 'w') as f:
                    f.write(text_content)
                print(f"Saved full text content to ipfs_content_{ipfs_hash}.txt")
                
                return text_content
            except UnicodeDecodeError:
                print("Not valid UTF-8 text")
                
            # If all else fails, return hex representation and save binary
            hex_sample = content[:100].hex()
            print(f"Unable to decode content. First 100 bytes (hex):\n{hex_sample}")
            
            # Save to a file for manual inspection
            file_path = f"ipfs_content_{ipfs_hash}.bin"
            with open(file_path, 'wb') as f:
                f.write(content)
            print(f"Saved binary content to {file_path} for manual inspection")
            
            return None
                    
        except Exception as e:
            print(f"Error inspecting content for hash {ipfs_hash}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def add_json(self, json_data: Dict[str, Any]) -> str:
        """
        Add JSON data to IPFS
        
        Args:
            json_data: JSON serializable data
            
        Returns:
            IPFS hash (CID) of the added JSON
        """
        # Convert data to JSON string
        json_str = json.dumps(json_data)
        
        # Add to IPFS as a file
        ipfs_hash = self.add_file(json_str.encode('utf-8'), "data.json")
        
        logger.info(f"Added JSON data to IPFS with hash: {ipfs_hash}")
        return ipfs_hash

    def get_json(self, ipfs_hash: str) -> Dict[str, Any]:
        """
        Get JSON data from IPFS
        
        Args:
            ipfs_hash: IPFS hash (CID) of the JSON data
            
        Returns:
            Parsed JSON data
        """
        # Get file content
        content = self.get_file(ipfs_hash)
        
        # Parse JSON
        try:
            json_data = json.loads(content.decode('utf-8'))
            logger.info(f"Retrieved and parsed JSON data from IPFS hash: {ipfs_hash}")
            return json_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from IPFS hash {ipfs_hash}: {e}")
            raise
    
    def upload_ensemble_state(self, ensemble_state: Dict[str, Any], extra_info: Dict[str, Any] = None) -> str:
        """
        Upload an ensemble state to IPFS
        
        Args:
            ensemble_state: Dictionary with ensemble state (weights, model_names, etc.)
            extra_info: Additional information to include
            
        Returns:
            IPFS hash of the uploaded ensemble state
        """
        # Create a data structure with both ensemble state and extra info
        data = {
            "ensemble_state": ensemble_state,
            "info": extra_info or {}
        }
        
        # Add to IPFS
        ipfs_hash = self.add_json(data)
        
        # Pin the file
        self.pin_file(ipfs_hash)
        
        logger.info(f"Uploaded ensemble state to IPFS with hash: {ipfs_hash}")
        return ipfs_hash
    
    def download_ensemble_state(self, ipfs_hash: str) -> Dict[str, Any]:
        """
        Download an ensemble state from IPFS
        
        Args:
            ipfs_hash: IPFS hash of the ensemble state
            
        Returns:
            Dictionary with the ensemble state
        """
        try:
            # Get the data from IPFS
            data = self.get_json(ipfs_hash)
            
            # Extract the ensemble state
            ensemble_state = data.get("ensemble_state", {})
            
            logger.info(f"Successfully downloaded ensemble state from IPFS: {ipfs_hash}")
            return ensemble_state
        except Exception as e:
            logger.error(f"Error downloading ensemble state from IPFS: {e}")
            raise