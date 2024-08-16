import os
import json
from joblib import dump
from sklearn.linear_model import SGDRegressor

def save_model(model, hyperparameters, metrics, folder="models/regression/linear_regression"):
    """
    Save the model, its hyperparameters, and its performance metrics to the specified folder.

    Args:
    - model: The trained model to be saved.
    - hyperparameters: Dictionary of the best hyperparameters.
    - metrics: Dictionary of the model's performance metrics.
    - folder: The folder where the files should be saved (default is "models/regression/linear_regression").
    """
    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Define the paths for the files
    model_path = os.path.join(folder, "model.joblib")
    hyperparameters_path = os.path.join(folder, "hyperparameters.json")
    metrics_path = os.path.join(folder, "metrics.json")

    # Save the model using joblib
    dump(model, model_path)

    # Save the hyperparameters to a JSON file
    with open(hyperparameters_path, 'w') as hp_file:
        json.dump(hyperparameters, hp_file, indent=4)

    # Save the metrics to a JSON file
    with open(metrics_path, 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    print(f"Model, hyperparameters, and metrics saved to {folder}")

# Hypothetical best model, hyperparameters, and metrics after tuning
best_model = SGDRegressor(alpha=0.001, max_iter=10000, penalty='l2')
best_hyperparameters = {
    'alpha': 0.001,
    'max_iter': 10000,
    'penalty': 'l2'
}
performance_metrics = {
    'validation_RMSE': 10.0,
    'test_RMSE': 12.0,
    'test_R2': 0.85
}

# Save the model, hyperparameters, and metrics
save_model(best_model, best_hyperparameters, performance_metrics)
