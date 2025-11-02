FEDERATED_CONFIG = {
    "num_clients": 10,
    "num_rounds": 100,
    "local_epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.01,
    "model_type": "cnn",
    "aggregation_method": "fedavg",
    "client_selection_ratio": 0.4,
    "secure_aggregation": True,
    "differential_privacy": True,
    "dp_epsilon": 1.0,
    "dp_delta": 1e-5,
    "data_distribution": "non-iid",
    "non_iid_alpha": 0.5,
    "min_clients_per_round": 3,
    "evaluation_frequency": 5,
    "save_checkpoint_frequency": 10
}

DATASET_CONFIG = {
    "name": "MNIST",
    "num_classes": 10,
    "input_shape": [1, 28, 28],
    "train_size": 60000,
    "test_size": 10000,
    "normalize": True
}

SECURITY_CONFIG = {
    "homomorphic_encryption": False,
    "secure_multi_party_computation": False,
    "model_encryption": True,
    "client_authentication": True,
    "audit_logging": True
}

COMMUNICATION_CONFIG = {
    "compression": True,
    "encryption": True,
    "max_retries": 3,
    "timeout": 30,
    "batch_updates": True
}