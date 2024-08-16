from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error
import pandas as pd
import numpy as np
from tabular_data import load_airbnb  # Import the load_airbnb function

# Load the Airbnb dataset
data = pd.read_csv('clean_tabular_data.csv')

# Extract features and labels using the imported load_airbnb function
features, labels = load_airbnb(data, label='Price_Night')

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Impute missing values in the training, validation, and test sets
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# Combine training and validation data to use with GridSearchCV
X_train_final = np.vstack([X_train_imputed, X_val_imputed])
y_train_final = np.hstack([y_train, y_val])

# Function to tune hyperparameters using GridSearchCV
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

# Define the hyperparameters to tune
param_grid = {
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'max_iter': [6000, 8000, 10000]
}

# Call the function to tune hyperparameters using GridSearchCV
best_model, best_hyperparameters, best_performance_metrics = tune_regression_model_hyperparameters(
    SGDRegressor, X_train_final, y_train_final, X_test_imputed, y_test, param_grid)

# Print the results
print("Best Model:", best_model)
print("\nBest Hyperparameters:", best_hyperparameters)
print("\nBest Performance Metrics:")
for metric, value in best_performance_metrics.items():
    print(metric + ":", value)
