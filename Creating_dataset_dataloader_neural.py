import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

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
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FullyConnectedNN, self).__init__()
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Training function
def train_and_evaluate(model, train_loader, val_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

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
    data_path = 'AirBnbData.csv'
    target_column = 'Price_Night'  
    num_epochs = 10

    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(data_path, target_column)
    
    # Initialize model
    input_dim = 17  
    hidden_dims = [512, 256, 128]  # Example hidden layer dimensions
    output_dim = 1  # Output dimension for regression (price per night)
    
    model = FullyConnectedNN(input_dim, hidden_dims, output_dim)
    
    # Train and evaluate the model
    train_and_evaluate(model, train_loader, val_loader, num_epochs)
