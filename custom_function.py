import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

def custom_tune_regression_model_hyperparameters(
    model_class,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    hyperparameters
):
    """
    Perform a grid search over hyperparameter values to find the best model.

    Args:
    - model_class: The model class (e.g., SGDRegressor, LinearRegression).
    - X_train: Training feature set.
    - y_train: Training labels.
    - X_val: Validation feature set.
    - y_val: Validation labels.
    - X_test: Test feature set.
    - y_test: Test labels.
    - hyperparameters: Dictionary of hyperparameter names mapping to a list of values to try.

    Returns:
    - best_model: The model instance with the best hyperparameters.
    - best_hyperparameters: Dictionary of the best hyperparameters.
    - performance_metrics: Dictionary of performance metrics.
    """

    best_model = None
    best_hyperparameters = None
    best_validation_rmse = float('inf')
    performance_metrics = {}

    # Get all combinations of hyperparameter values
    hyperparameter_combinations = list(product(*hyperparameters.values()))
    hyperparameter_names = list(hyperparameters.keys())

    for combination in hyperparameter_combinations:
        # Create a dictionary of hyperparameters for the current combination
        params = dict(zip(hyperparameter_names, combination))
        
        # Initialize the model with the current hyperparameters
        model = model_class(**params)
        
        # Fit the model on the training data
        model.fit(X_train, y_train)
        
        # Predict on the validation set
        val_predictions = model.predict(X_val)
        
        # Calculate RMSE for the validation set
        val_rmse = mean_squared_error(y_val, val_predictions, squared=False)
        
        # Check if this model is the best so far
        if val_rmse < best_validation_rmse:
            best_validation_rmse = val_rmse
            best_model = model
            best_hyperparameters = params
            
    # Once the best model is found, evaluate it on the test set
    test_predictions = best_model.predict(X_test)
    test_rmse = mean_squared_error(y_test, test_predictions, squared=False)
    test_r2 = r2_score(y_test, test_predictions)

    # Store the performance metrics
    performance_metrics['validation_RMSE'] = best_validation_rmse
    performance_metrics['test_RMSE'] = test_rmse
    performance_metrics['test_R2'] = test_r2

    return best_model, best_hyperparameters, performance_metrics

# Create synthetic regression data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define hyperparameters to tune
hyperparameters = {
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [1000, 5000, 10000],
    'penalty': ['l2', 'l1', 'elasticnet']
}

# Perform custom grid search
best_model, best_hyperparameters, performance_metrics = custom_tune_regression_model_hyperparameters(
    SGDRegressor,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    hyperparameters
)

# Display results
print("Best Hyperparameters:", best_hyperparameters)
print("Performance Metrics:", performance_metrics)
