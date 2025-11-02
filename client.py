import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import copy
import time
import hashlib

class FederatedClient:
    def __init__(self, client_id, local_model, train_loader, test_loader, config):
        self.client_id = client_id
        self.local_model = local_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        
        self.local_epochs = config.get('local_epochs', 5)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.batch_size = config.get('batch_size', 32)
        
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.training_history = []
        self.model_versions = {}
        
    def local_train(self, global_state_dict):
        self.local_model.load_state_dict(global_state_dict)
        
        original_state = copy.deepcopy(self.local_model.state_dict())
        
        self.local_model.train()
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                
        trained_state = self.local_model.state_dict()
        
        update = {}
        for key in trained_state.keys():
            update[key] = trained_state[key] - original_state[key]
            
        num_samples = len(self.train_loader.dataset)
        
        metadata = {
            'client_id': self.client_id,
            'training_time': time.time(),
            'num_samples': num_samples,
            'average_loss': epoch_loss / len(self.train_loader),
            'model_hash': self._compute_model_hash(trained_state)
        }
        
        self.training_history.append({
            'round': len(self.training_history) + 1,
            'timestamp': time.time(),
            'loss': epoch_loss / len(self.train_loader),
            'samples_used': num_samples
        })
        
        return update, num_samples, metadata
    
    def local_train_fedprox(self, global_state_dict, mu: float = 0.01):
        self.local_model.load_state_dict(global_state_dict)
        global_model_copy = copy.deepcopy(self.local_model)
        
        original_state = copy.deepcopy(self.local_model.state_dict())
        
        self.local_model.train()
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                
                proximal_term = 0.0
                for param_local, param_global in zip(self.local_model.parameters(), 
                                                   global_model_copy.parameters()):
                    proximal_term += torch.norm(param_local - param_global) ** 2
                    
                loss += (mu / 2) * proximal_term
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                
        trained_state = self.local_model.state_dict()
        
        update = {}
        for key in trained_state.keys():
            update[key] = trained_state[key] - original_state[key]
            
        num_samples = len(self.train_loader.dataset)
        
        metadata = {
            'client_id': self.client_id,
            'training_time': time.time(),
            'num_samples': num_samples,
            'average_loss': epoch_loss / len(self.train_loader),
            'model_hash': self._compute_model_hash(trained_state),
            'method': 'fedprox'
        }
        
        return update, num_samples, metadata
    
    def evaluate_local_model(self):
        self.local_model.eval()
        correct = 0
        total = 0
        total_loss = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.local_model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.test_loader)
        
        return accuracy, avg_loss
    
    def _compute_model_hash(self, state_dict):
        model_bytes = str(sorted(state_dict.items())).encode()
        return hashlib.sha256(model_bytes).hexdigest()
    
    def save_local_checkpoint(self, path: str):
        checkpoint = {
            'local_model_state': self.local_model.state_dict(),
            'training_history': self.training_history,
            'client_id': self.client_id,
            'config': self.config
        }
        torch.save(checkpoint, path)
        
    def load_local_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.local_model.load_state_dict(checkpoint['local_model_state'])
        self.training_history = checkpoint['training_history']
        self.client_id = checkpoint['client_id']