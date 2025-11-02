import torch
import torchvision
import torchvision.transforms as transforms
from trainer import FederatedTrainer
from config import FEDERATED_CONFIG
import os

def setup_directories():
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

def load_mnist_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    return train_dataset, test_dataset

def main():
    setup_directories()
    
    print("Loading MNIST Dataset...")
    train_dataset, test_dataset = load_mnist_dataset()
    
    config = FEDERATED_CONFIG.copy()
    config['num_classes'] = 10
    
    print("Initializing Federated Trainer...")
    trainer = FederatedTrainer(config, train_dataset, test_dataset)
    
    print("Starting Federated Learning...")
    trainer.train_federated()
    
    print("Evaluating Client Models...")
    client_results = trainer.evaluate_client_models()
    
    print("\nClient Evaluation Results:")
    for client_id, results in client_results.items():
        print(f"{client_id}: Accuracy = {results['accuracy']:.2f}%, "
              f"Loss = {results['loss']:.4f}, "
              f"Samples = {results['num_samples']}")
              
    print(f"\nFederated Learning completed successfully!")
    print(f"Best global model accuracy: {trainer.best_accuracy:.2f}%")

if __name__ == "__main__":
    main()