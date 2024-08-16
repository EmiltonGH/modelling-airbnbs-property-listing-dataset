import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from tabular_data import load_airbnb 

# Load data
data = pd.read_csv('clean_tabular_data.csv')  

# Encode categorical labels
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])

# Load the features and labels
X, y = load_airbnb(data, label='Category')

print("\nFeature data types before handling missing values:")
print(X.dtypes)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def tune_classification_model_hyperparameters(model_class, X_train, y_train, X_test, y_test, param_grid):
    """
    Tunes the hyperparameters of a classification model using GridSearchCV.
    """
    # Create a pipeline that includes imputation, scaling, and the model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler()),  # Feature scaling
        ('model', model_class(max_iter=5000))  # Increase max_iter further
    ])

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    # Evaluate performance on the validation set
    validation_accuracy = grid_search.best_score_

    # Evaluate performance on the test set
    test_predictions = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Prepare the performance metrics
    performance_metrics = {
        'validation_accuracy': validation_accuracy,
        'test_accuracy': test_accuracy
    }

    return best_model, best_hyperparameters, performance_metrics

# Define the hyperparameter grid
param_grid = {
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Wider range of C values
    'model__solver': ['liblinear', 'lbfgs', 'saga']  # Keep solver options
}

# Tune the logistic regression model
best_model, best_hyperparameters, performance_metrics = tune_classification_model_hyperparameters(
    model_class=LogisticRegression,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    param_grid=param_grid
)

# Output the results
print("Best Hyperparameters:", best_hyperparameters)
print("Performance Metrics:", performance_metrics)

def evaluate_model_performance(model, X_train, y_train, X_test, y_test):
    """
    Evaluate and print the model's performance metrics.
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    print("Training Set Performance:")
    print(f"Accuracy: {train_accuracy}")
    print(f"Precision: {train_precision}")
    print(f"Recall: {train_recall}")
    print(f"F1 Score: {train_f1}")
    print("\nTest Set Performance:")
    print(f"Accuracy: {test_accuracy}")
    print(f"Precision: {test_precision}")
    print(f"Recall: {test_recall}")
    print(f"F1 Score: {test_f1}")

# Evaluate the tuned model
evaluate_model_performance(best_model, X_train, y_train, X_test, y_test)
