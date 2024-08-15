import os
import json
import torch
import time
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import yaml

# YAML configuration (create nn_config.yaml with the following content)
"""
# nn_config.yaml
optimiser: Adam
learning_rate: 0.001
hidden_layer_width: 256
depth: 3
"""

# Function to read the YAML configuration
def get_nn_config(config_file='nn_config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Dataset class
class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self, data, target_column):
        # Separate features and target
        self.features = data.drop(columns=[target_column, 'ID', 'Title', 'Description', 'Amenities', 'url', 'Unnamed: 19'])
        self.labels = data[target_column]
        
        # Identify categorical and numerical columns
        categorical_features = ['Category', 'Location']  # List your categorical columns here
        numerical_features = self.features.select_dtypes(include=[np.number]).columns.tolist()

        # Remove non-numeric columns
        self.features = self.features[numerical_features + categorical_features]

        # Create preprocessing pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Ensure dense output
                ]), categorical_features)
            ])
        
        # Fit and transform features
        self.features = preprocessor.fit_transform(self.features)

        # Print the shape of the features to debug
        print("Shape of features after preprocessing:", self.features.shape)
        
        # Convert features and labels to PyTorch tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Fully connected neural network model
class FullyConnectedNN(nn.Module):
    def __init__(self, config):
        super(FullyConnectedNN, self).__init__()
        hidden_layer_width = config['hidden_layer_width']
        depth = config['depth']
        input_dim = config['input_dim']
        output_dim = 1  # Assuming regression, output is a single value

        layers = []
        in_dim = input_dim

        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_layer_width))
            layers.append(nn.ReLU())
            in_dim = hidden_layer_width

        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Function to evaluate the model
def evaluate_model(model, loader, criterion):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_labels = []
    all_predictions = []
    total_loss = 0.0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.squeeze().cpu().numpy())

    avg_loss = np.sqrt(total_loss / len(loader))
    r_squared = r2_score(all_labels, all_predictions)
    return avg_loss, r_squared

# Training function
def train_and_evaluate(model, train_loader, val_loader, test_loader, config, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()

    optimizer_name = config['optimiser']
    learning_rate = config['learning_rate']
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

    training_duration = time.time() - start_time

    # Evaluation
    train_loss, train_r2 = evaluate_model(model, train_loader, criterion)
    val_loss, val_r2 = evaluate_model(model, val_loader, criterion)
    test_loss, test_r2 = evaluate_model(model, test_loader, criterion)

    # Inference latency calculation
    num_inference_samples = 100  
    inference_start_time = time.time()
    for _ in range(num_inference_samples):
        _ = model(train_loader.dataset[0][0].unsqueeze(0).to(device))
    inference_latency = (time.time() - inference_start_time) / num_inference_samples

    # Prepare metrics
    metrics = {
        'RMSE_loss': {
            'train': train_loss,
            'validation': val_loss,
            'test': test_loss,
        },
        'R_squared': {
            'train': train_r2,
            'validation': val_r2,
            'test': test_r2,
        },
        'training_duration': training_duration,
        'inference_latency': inference_latency,
    }

    save_model(model, config, train_loader, val_loader, test_loader, metrics)

# Function to save the model, hyperparameters, and metrics
def save_model(model, config, train_loader, val_loader, test_loader, metrics):
    # Create the directory structure
    base_dir = 'models/neural_networks/regression'
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_dir = os.path.join(base_dir, timestamp)
    os.makedirs(model_dir, exist_ok=True)

    # Save the model if it's a PyTorch model
    if isinstance(model, torch.nn.Module):
        model_path = os.path.join(model_dir, 'model.pt')
        torch.save(model.state_dict(), model_path)
        
        # Save the hyperparameters to a JSON file
        hyperparameters_path = os.path.join(model_dir, 'hyperparameters.json')
        with open(hyperparameters_path, 'w') as f:
            json.dump(config, f, indent=4)

        # Save the metrics to a JSON file
        metrics_path = os.path.join(model_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Model, hyperparameters, and metrics saved in: {model_dir}")
    else:
        raise ValueError("The provided model is not a PyTorch module")

# Prepare dataloaders
def prepare_dataloaders(data_path, target_column, batch_size=64):
    data = pd.read_csv(data_path)
    
    # Create the dataset
    dataset = AirbnbNightlyPriceRegressionDataset(data, target_column)

    # Split the dataset into training, validation, and test sets
    num_samples = len(dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    
    train_end = int(0.7 * num_samples)
    val_end = int(0.85 * num_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Main execution
if __name__ == "__main__":
    data_path = 'AirBnbData.csv'  # Update with your dataset path
    target_column = 'Price_Night'  # Update with your target column
    num_epochs = 10
    
    # Load configuration from YAML file
    config = get_nn_config()

    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(data_path, target_column)

    # Add input_dim to the config based on the preprocessed data
    input_dim = train_loader.dataset[0][0].shape[0]
    config['input_dim'] = input_dim

    # Initialize model with config
    model = FullyConnectedNN(config)

    # Train and evaluate the model
    train_and_evaluate(model, train_loader, val_loader, test_loader, config, num_epochs)
