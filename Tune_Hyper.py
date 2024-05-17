from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from tabular_data import load_airbnb
from sklearn.metrics import mean_squared_error
from itertools import product
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparameters):
    best_model = None
    best_hyperparameters = {}
    best_performance_metrics = {"validation_RMSE": float('inf')}
    
    # Iterate over hyperparameter combinations
    for hyperparameter_combination in generate_hyperparameter_combinations(hyperparameters):
        # Initialize model with hyperparameters
        model = model_class(**hyperparameter_combination)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_predictions = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        
        # Update best model and hyperparameters if current model is better
        if val_rmse < best_performance_metrics["validation_RMSE"]:
            best_model = model
            best_hyperparameters = hyperparameter_combination
            best_performance_metrics["validation_RMSE"] = val_rmse
    
    # Evaluate best model on test set
    test_predictions = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    best_performance_metrics["test_RMSE"] = test_rmse
    
    return best_model, best_hyperparameters, best_performance_metrics

def generate_hyperparameter_combinations(hyperparameters):
    keys = hyperparameters.keys()
    values = hyperparameters.values()
    for combination in product(*values):
        yield dict(zip(keys, combination))

# Load the Airbnb dataset
data = pd.read_csv('clean_tabular_data.csv')  

# Extract features and labels
features, labels = load_airbnb(data, label='Price_Night')

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the hyperparameters to tune
hyperparameters = {
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'max_iter': [1000, 2000, 3000]
}

# Impute missing values in the training, validation, and test sets
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# Call the function to tune hyperparameters with imputed data
best_model, best_hyperparameters, best_performance_metrics = custom_tune_regression_model_hyperparameters(SGDRegressor, X_train_imputed, y_train, X_val_imputed, y_val, X_test_imputed, y_test, hyperparameters)

# Print the results
print("Best Model:", best_model)
print("\nBest Hyperparameters:", best_hyperparameters)
print("\nBest Performance Metrics:")
for metric, value in best_performance_metrics.items():
    print(metric + ":", value)
