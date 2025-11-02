import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import json
import pickle
import hashlib
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class FederatedDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def split_data_iid(dataset, num_clients, samples_per_client=None):
    num_samples = len(dataset)
    if samples_per_client is None:
        samples_per_client = num_samples // num_clients
        
    client_datasets = []
    indices = np.random.permutation(num_samples)
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        if i == num_clients - 1:
            end_idx = num_samples
            
        client_indices = indices[start_idx:end_idx]
        client_dataset = Subset(dataset, client_indices)
        client_datasets.append(client_dataset)
        
    return client_datasets

def split_data_non_iid(dataset, num_clients, num_classes, alpha=0.5):
    client_datasets = []
    labels = np.array([target for _, target in dataset])
    
    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < len(labels) / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            
        min_size = min(len(idx_j) for idx_j in idx_batch)
        
    for i in range(num_clients):
        client_dataset = Subset(dataset, idx_batch[i])
        client_datasets.append(client_dataset)
        
    return client_datasets

def add_differential_privacy(parameters, epsilon, delta, sensitivity):
    for param in parameters:
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = torch.randn_like(param) * noise_scale
        param.add_(noise)
        
    return parameters

def compute_model_similarity(model1, model2):
    similarity_score = 0.0
    total_params = 0
    
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 == name2:
            similarity = torch.cosine_similarity(param1.flatten(), param2.flatten(), dim=0)
            similarity_score += similarity.item() * param1.numel()
            total_params += param1.numel()
            
    return similarity_score / total_params

def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    rounds = [entry['round'] for entry in history]
    clients = [entry['clients_participated'] for entry in history]
    
    ax1.plot(rounds, clients, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Clients Participated')
    ax1.set_title('Client Participation Over Rounds')
    ax1.grid(True, alpha=0.3)
    
    samples = [entry['total_samples'] for entry in history]
    ax2.plot(rounds, samples, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Total Samples')
    ax2.set_title('Total Training Samples Per Round')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_federated_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def load_federated_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_accuracy(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    return 100 * correct / total

def encrypt_model_updates(model_updates, key):
    import hashlib
    from cryptography.fernet import Fernet
    
    fernet = Fernet(key)
    encrypted_updates = {}
    
    for client_id, update in model_updates.items():
        serialized_update = pickle.dumps(update)
        encrypted_update = fernet.encrypt(serialized_update)
        encrypted_updates[client_id] = encrypted_update
        
    return encrypted_updates

def decrypt_model_updates(encrypted_updates, key):
    from cryptography.fernet import Fernet
    
    fernet = Fernet(key)
    decrypted_updates = {}
    
    for client_id, encrypted_update in encrypted_updates.items():
        decrypted_data = fernet.decrypt(encrypted_update)
        update = pickle.loads(decrypted_data)
        decrypted_updates[client_id] = update
        
    return decrypted_updates