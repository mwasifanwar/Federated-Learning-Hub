import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import json
import time
from typing import Dict, List, Tuple
import hashlib

class FederatedServer:
    def __init__(self, global_model, config):
        self.global_model = global_model
        self.config = config
        self.client_models = {}
        self.client_weights = {}
        self.client_metadata = {}
        self.training_history = []
        self.model_versions = {}
        
        self.secure_aggregation_enabled = config.get('secure_aggregation', True)
        self.differential_privacy = config.get('differential_privacy', True)
        self.dp_epsilon = config.get('dp_epsilon', 1.0)
        self.dp_delta = config.get('dp_delta', 1e-5)
        
    def initialize_global_model(self):
        model_state = self.global_model.state_dict()
        torch.save(model_state, 'models/global_model_initial.pth')
        
    def receive_client_update(self, client_id: str, model_state: Dict, 
                            num_samples: int, metadata: Dict):
        self.client_models[client_id] = copy.deepcopy(model_state)
        self.client_weights[client_id] = num_samples
        self.client_metadata[client_id] = metadata
        
    def aggregate_updates_fedavg(self) -> Dict:
        total_samples = sum(self.client_weights.values())
        global_state = self.global_model.state_dict()
        
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key])
            
        for client_id, client_state in self.client_models.items():
            weight = self.client_weights[client_id] / total_samples
            
            for key in global_state.keys():
                if self.differential_privacy:
                    noise_scale = self._calculate_noise_scale(weight)
                    noise = torch.randn_like(client_state[key]) * noise_scale
                    client_state[key] += noise
                    
                global_state[key] += client_state[key] * weight
                
        return global_state
    
    def aggregate_updates_fedprox(self, mu: float = 0.01) -> Dict:
        total_samples = sum(self.client_weights.values())
        global_state = self.global_model.state_dict()
        current_global_state = copy.deepcopy(global_state)
        
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key])
            
        for client_id, client_state in self.client_models.items():
            weight = self.client_weights[client_id] / total_samples
            
            for key in global_state.keys():
                if self.differential_privacy:
                    noise_scale = self._calculate_noise_scale(weight)
                    noise = torch.randn_like(client_state[key]) * noise_scale
                    client_state[key] += noise
                    
                proximal_term = mu * (client_state[key] - current_global_state[key])
                global_state[key] += (client_state[key] - proximal_term) * weight
                
        return global_state
    
    def _calculate_noise_scale(self, weight: float) -> float:
        sensitivity = 2.0 * weight
        epsilon = self.dp_epsilon
        delta = self.dp_delta
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        return noise_scale
    
    def secure_aggregation(self, client_updates: List[Dict]) -> Dict:
        if not self.secure_aggregation_enabled:
            return self.aggregate_updates_fedavg()
            
        masked_updates = []
        for update in client_updates:
            mask = torch.randn_like(list(update.values())[0])
            masked_update = {k: v + mask for k, v in update.items()}
            masked_updates.append(masked_update)
            
        aggregated = {}
        for key in masked_updates[0].keys():
            aggregated[key] = torch.stack([update[key] for update in masked_updates]).mean(dim=0)
            
        return aggregated
    
    def update_global_model(self, aggregation_method: str = 'fedavg'):
        if aggregation_method == 'fedavg':
            new_global_state = self.aggregate_updates_fedavg()
        elif aggregation_method == 'fedprox':
            new_global_state = self.aggregate_updates_fedprox()
        else:
            new_global_state = self.aggregate_updates_fedavg()
            
        self.global_model.load_state_dict(new_global_state)
        
        training_round = len(self.training_history) + 1
        self.training_history.append({
            'round': training_round,
            'timestamp': time.time(),
            'clients_participated': len(self.client_models),
            'total_samples': sum(self.client_weights.values())
        })
        
        self.client_models.clear()
        self.client_weights.clear()
        
    def evaluate_global_model(self, test_loader):
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.global_model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return accuracy, avg_loss
    
    def save_checkpoint(self, path: str):
        checkpoint = {
            'global_model_state': self.global_model.state_dict(),
            'training_history': self.training_history,
            'config': self.config,
            'model_versions': self.model_versions
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.global_model.load_state_dict(checkpoint['global_model_state'])
        self.training_history = checkpoint['training_history']
        self.config = checkpoint['config']
        self.model_versions = checkpoint['model_versions']