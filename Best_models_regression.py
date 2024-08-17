import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression
from joblib import dump, load
import os
import json

def save_model(model, hyperparameters, metrics, folder):
    os.makedirs(folder, exist_ok=True)
    dump(model, os.path.join(folder, 'model.joblib'))
    with open(os.path.join(folder, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f)
    with open(os.path.join(folder, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    print(f"Model saved to {folder}")

def tune_regression_model_hyperparameters(model_class, X_train, y_train, X_test, y_test, param_grid, cv=5):
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    grid_search = GridSearchCV(model_class(), param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_
    
    test_predictions = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    
    best_performance_metrics = {
        "validation_RMSE": np.sqrt(-grid_search.best_score_),
        "test_RMSE": test_rmse
    }
    
    return best_model, best_hyperparameters, best_performance_metrics

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
            model_class, X_train, y_train, X_test, y_test, param_grids[model_name]
        )

        test_predictions = best_model.predict(X_test)
        test_rmse = mean_squared_error(y_test, test_predictions, squared=False)
        test_r2 = r2_score(y_test, test_predictions)

        metrics = {
            'validation_RMSE': best_score["validation_RMSE"],
            'test_RMSE': test_rmse,
            'test_R2': test_r2,
        }

        save_model(best_model, best_hyperparameters, metrics, folder=f"models/regression/{model_name}")

def find_best_model():
    model_dirs = ["models/regression/decision_tree", 
                  "models/regression/random_forest", 
                  "models/regression/gradient_boosting"]
    
    best_model = None
    best_hyperparameters = None
    best_metrics = None
    best_rmse = float('inf')

    for model_dir in model_dirs:
        # Load the model, hyperparameters, and metrics
        model = load(os.path.join(model_dir, 'model.joblib'))
        with open(os.path.join(model_dir, 'hyperparameters.json'), 'r') as f:
            hyperparameters = json.load(f)
        with open(os.path.join(model_dir, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        
        # Compare RMSE
        if metrics['test_RMSE'] < best_rmse:
            best_rmse = metrics['test_RMSE']
            best_model = model
            best_hyperparameters = hyperparameters
            best_metrics = metrics

    return best_model, best_hyperparameters, best_metrics

if __name__ == "__main__":
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Evaluate and save all models
    evaluate_all_models(X_train, y_train, X_test, y_test)
    
    # Find the best model
    best_model, best_hyperparameters, best_metrics = find_best_model()
    
    # Print the best model details
    print("\nBest Model Details:")
    print("Best Hyperparameters:", best_hyperparameters)
    print("Best Metrics:", best_metrics)
