import hashlib
import hmac
import secrets
from cryptography.fernet import Fernet
import json
import time
import torch

class SecurityManager:
    def __init__(self):
        self.client_keys = {}
        self.session_tokens = {}
        
    def generate_client_key(self, client_id):
        key = Fernet.generate_key()
        self.client_keys[client_id] = key
        return key
    
    def encrypt_model_update(self, client_id, model_update):
        if client_id not in self.client_keys:
            raise ValueError(f"No key found for client {client_id}")
            
        fernet = Fernet(self.client_keys[client_id])
        serialized_update = json.dumps(self._serialize_model_update(model_update)).encode()
        encrypted_update = fernet.encrypt(serialized_update)
        
        return encrypted_update
    
    def decrypt_model_update(self, client_id, encrypted_update):
        if client_id not in self.client_keys:
            raise ValueError(f"No key found for client {client_id}")
            
        fernet = Fernet(self.client_keys[client_id])
        decrypted_data = fernet.decrypt(encrypted_update)
        model_update = self._deserialize_model_update(json.loads(decrypted_data.decode()))
        
        return model_update
    
    def _serialize_model_update(self, model_update):
        serialized = {}
        for key, tensor in model_update.items():
            serialized[key] = {
                'data': tensor.cpu().numpy().tolist(),
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype)
            }
        return serialized
    
    def _deserialize_model_update(self, serialized_update):
        model_update = {}
        for key, tensor_info in serialized_update.items():
            array = np.array(tensor_info['data'])
            tensor = torch.tensor(array).reshape(tensor_info['shape'])
            model_update[key] = tensor
        return model_update
    
    def generate_session_token(self, client_id, expiry_hours=24):
        token = secrets.token_hex(32)
        expiry = time.time() + (expiry_hours * 3600)
        self.session_tokens[token] = {
            'client_id': client_id,
            'expiry': expiry
        }
        return token
    
    def validate_session_token(self, token):
        if token not in self.session_tokens:
            return False
            
        session = self.session_tokens[token]
        if time.time() > session['expiry']:
            del self.session_tokens[token]
            return False
            
        return True
    
    def create_hmac_signature(self, data, secret_key):
        message = json.dumps(data, sort_keys=True).encode()
        signature = hmac.new(secret_key.encode(), message, hashlib.sha256).hexdigest()
        return signature
    
    def verify_hmac_signature(self, data, signature, secret_key):
        expected_signature = self.create_hmac_signature(data, secret_key)
        return hmac.compare_digest(signature, expected_signature)

import numpy as np