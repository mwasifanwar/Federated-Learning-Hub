import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import json
import os
from server import FederatedServer
from client import FederatedClient
from models import create_model
from utils import split_data_iid, split_data_non_iid, compute_accuracy, plot_training_history

class FederatedTrainer:
    def __init__(self, config, train_dataset, test_dataset):
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        self.global_model = create_model(
            config['model_type'], 
            num_classes=config.get('num_classes', 10)
        )
        
        self.server = FederatedServer(self.global_model, config)
        self.clients = []
        self.setup_clients()
        
        self.training_history = []
        self.best_accuracy = 0.0
        
    def setup_clients(self):
        num_clients = self.config['num_clients']
        data_distribution = self.config.get('data_distribution', 'iid')
        
        if data_distribution == 'iid':
            client_datasets = split_data_iid(self.train_dataset, num_clients)
        else:
            alpha = self.config.get('non_iid_alpha', 0.5)
            num_classes = self.config.get('num_classes', 10)
            client_datasets = split_data_non_iid(
                self.train_dataset, num_clients, num_classes, alpha
            )
            
        for i in range(num_clients):
            client_train_loader = DataLoader(
                client_datasets[i], 
                batch_size=self.config['batch_size'], 
                shuffle=True
            )
            
            client_test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False
            )
            
            client_model = create_model(
                self.config['model_type'],
                num_classes=self.config.get('num_classes', 10)
            )
            
            client = FederatedClient(
                client_id=f"client_{i}",
                local_model=client_model,
                train_loader=client_train_loader,
                test_loader=client_test_loader,
                config=self.config
            )
            
            self.clients.append(client)
            
    def select_clients_for_round(self, round_num):
        total_clients = len(self.clients)
        selection_ratio = self.config.get('client_selection_ratio', 0.4)
        num_selected = max(int(total_clients * selection_ratio), 
                          self.config.get('min_clients_per_round', 1))
        
        selected_indices = np.random.choice(
            total_clients, num_selected, replace=False
        )
        
        return [self.clients[i] for i in selected_indices]
    
    def train_federated(self):
        num_rounds = self.config['num_rounds']
        aggregation_method = self.config.get('aggregation_method', 'fedavg')
        
        print("Starting Federated Training...")
        print(f"Total Rounds: {num_rounds}")
        print(f"Clients: {len(self.clients)}")
        print(f"Aggregation: {aggregation_method}")
        
        self.server.initialize_global_model()
        
        for round_num in range(1, num_rounds + 1):
            round_start_time = time.time()
            
            selected_clients = self.select_clients_for_round(round_num)
            global_state = self.server.global_model.state_dict()
            
            print(f"\nRound {round_num}/{num_rounds}")
            print(f"Selected {len(selected_clients)} clients")
            
            for client in selected_clients:
                if aggregation_method == 'fedprox':
                    update, num_samples, metadata = client.local_train_fedprox(
                        global_state, mu=0.01
                    )
                else:
                    update, num_samples, metadata = client.local_train(global_state)
                    
                self.server.receive_client_update(
                    client.client_id, update, num_samples, metadata
                )
                
            self.server.update_global_model(aggregation_method)
            
            if round_num % self.config.get('evaluation_frequency', 5) == 0:
                accuracy, loss = self.server.evaluate_global_model(self.test_loader)
                self.training_history.append({
                    'round': round_num,
                    'accuracy': accuracy,
                    'loss': loss,
                    'clients_participated': len(selected_clients),
                    'timestamp': time.time()
                })
                
                print(f"Round {round_num} - Test Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
                
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.server.save_checkpoint(f'models/best_model_round_{round_num}.pth')
                    
            if round_num % self.config.get('save_checkpoint_frequency', 10) == 0:
                self.server.save_checkpoint(f'models/checkpoint_round_{round_num}.pth')
                
            round_time = time.time() - round_start_time
            print(f"Round {round_num} completed in {round_time:.2f} seconds")
            
        final_accuracy, final_loss = self.server.evaluate_global_model(self.test_loader)
        print(f"\nFinal Results - Accuracy: {final_accuracy:.2f}%, Loss: {final_loss:.4f}")
        print(f"Best Accuracy: {self.best_accuracy:.2f}%")
        
        self.save_training_results()
        
    def save_training_results(self):
        results = {
            'config': self.config,
            'training_history': self.training_history,
            'best_accuracy': self.best_accuracy,
            'final_model_path': 'models/final_global_model.pth'
        }
        
        with open('results/training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        self.server.save_checkpoint('models/final_global_model.pth')
        
        plot_training_history(
            self.server.training_history, 
            'results/training_progress.png'
        )
        
    def evaluate_client_models(self):
        client_results = {}
        
        for client in self.clients:
            accuracy, loss = client.evaluate_local_model()
            client_results[client.client_id] = {
                'accuracy': accuracy,
                'loss': loss,
                'num_samples': len(client.train_loader.dataset)
            }
            
        return client_results