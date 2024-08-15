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
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error

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

# Fully connected neural network model with Dropout
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FullyConnectedNN, self).__init__()
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))  # Add Dropout
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Training function with early stopping
def train_and_evaluate(model, train_loader, val_loader, num_epochs, patience=3, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir='runs/airbnb_price_regression')

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)

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
        writer.add_scalar('Loss/validation', val_loss, epoch)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save model with the best validation loss
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break

    # Close the TensorBoard writer
    writer.close()

# Evaluation function
def evaluate(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

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
    learning_rate = 0.001

    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(data_path, target_column)
    
    # Initialize model
    input_dim = 777  # Update this to match the actual number of features
    hidden_dims = [512, 256, 128]  # Example hidden layer dimensions
    output_dim = 1  # Output dimension for regression (price per night)
    
    model = FullyConnectedNN(input_dim, hidden_dims, output_dim)
    
    # Train and evaluate the model
    train_and_evaluate(model, train_loader, val_loader, num_epochs, patience=3, learning_rate=learning_rate)

    # Evaluate on the test set
    test_rmse = evaluate(model, test_loader)
    print(f'Test RMSE: {test_rmse:.4f}')
