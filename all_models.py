import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn_hyperparameters import tune_regression_model_hyperparameters

import os
import json
from joblib import dump

# Define the save_model function
def save_model(model, hyperparameters, metrics, folder):
    os.makedirs(folder, exist_ok=True)
    dump(model, os.path.join(folder, 'model.joblib'))
    with open(os.path.join(folder, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f)
    with open(os.path.join(folder, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    print(f"Model saved to {folder}")

# Assuming the tune_regression_model_hyperparameters function is already defined above

def evaluate_all_models(X_train, y_train, X_test, y_test):
    param_grids = {
        'decision_tree': {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        'gradient_boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
        }
    }

    models = {
        'decision_tree': DecisionTreeRegressor,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
    }

    for model_name, model_class in models.items():
        print(f"Evaluating {model_name}...")

        best_model, best_hyperparameters, best_score = tune_regression_model_hyperparameters(
            model_class,
            param_grids[model_name],
            X_train,
            y_train
        )

        test_predictions = best_model.predict(X_test)
        test_rmse = mean_squared_error(y_test, test_predictions, squared=False)
        test_r2 = r2_score(y_test, test_predictions)

        metrics = {
            'validation_RMSE': best_score,
            'test_RMSE': test_rmse,
            'test_R2': test_r2,
        }

        save_model(best_model, best_hyperparameters, metrics, folder=f"models/regression/{model_name}")

if __name__ == "__main__":
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    evaluate_all_models(X_train, y_train, X_test, y_test)
