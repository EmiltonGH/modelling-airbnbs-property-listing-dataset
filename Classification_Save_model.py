import os
import json
import pandas as pd
from joblib import dump
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabular_data import load_airbnb  

def save_model(model, hyperparameters, metrics, folder="models/classification/logistic_regression"):
    """
    Save the model, its hyperparameters, and its performance metrics to the specified folder.
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

# Load data
data = pd.read_csv('clean_tabular_data.csv')

# Encode categorical labels
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])

# Load the features and labels
X, y = load_airbnb(data, label='Category')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for logistic regression
param_grid = {
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'model__solver': ['liblinear', 'lbfgs', 'saga']
}

# Define the function to tune the model and save it
def tune_and_save_model():
    # Create a pipeline with an imputer, scaler, and logistic regression model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=10000))
    ])

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    # Evaluate performance on the test set
    test_predictions = best_model.predict(X_test)
    performance_metrics = {
        'test_accuracy': accuracy_score(y_test, test_predictions),
        'test_precision': precision_score(y_test, test_predictions, average='weighted'),
        'test_recall': recall_score(y_test, test_predictions, average='weighted'),
        'test_f1': f1_score(y_test, test_predictions, average='weighted')
    }

    # Save the model, hyperparameters, and metrics
    save_model(best_model, best_hyperparameters, performance_metrics, folder="models/classification/logistic_regression")

# Call the function to tune and save the model
tune_and_save_model()
