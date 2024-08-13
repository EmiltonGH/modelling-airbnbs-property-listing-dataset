from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np

def tune_regression_model_hyperparameters(model_class, param_grid, X_train, y_train, cv=5):
    """
    Perform a grid search over hyperparameter values using GridSearchCV.

    Args:
    - model_class: The model class (e.g., SGDRegressor, LinearRegression).
    - param_grid: Dictionary with hyperparameter names mapping to a list of values to try.
    - X_train: Training feature set.
    - y_train: Training labels.
    - cv: Number of cross-validation folds (default is 5).

    Returns:
    - best_model: The model instance with the best hyperparameters found.
    - best_hyperparameters: Dictionary of the best hyperparameters found.
    - best_score: The best cross-validated score (lower RMSE) on the validation set.
    """

    # Initialize the model with default parameters
    model = model_class()

    # Define RMSE as the scoring metric for GridSearchCV
    rmse_scorer = make_scorer(mean_squared_error, squared=False, greater_is_better=False)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(model, param_grid, scoring=rmse_scorer, cv=cv, n_jobs=-1)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Extract the best model, hyperparameters, and the best score
    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_
    best_score = -grid_search.best_score_  # convert back to positive RMSE

    return best_model, best_hyperparameters, best_score

# Example usage
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Create synthetic regression data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters to tune
param_grid = {
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [1000, 5000, 10000],
    'penalty': ['l2', 'l1', 'elasticnet']
}

# Perform grid search using the function
best_model, best_hyperparameters, best_score = tune_regression_model_hyperparameters(
    SGDRegressor,
    param_grid,
    X_train,
    y_train
)

# Display results
print("Best Hyperparameters:", best_hyperparameters)
print("Best RMSE on validation set:", best_score)
