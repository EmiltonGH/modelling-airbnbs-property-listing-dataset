import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump, load

def save_model(model, hyperparameters, metrics, folder):
    """
    Save the model, its hyperparameters, and its performance metrics to the specified folder.

    Args:
    - model: The trained model to be saved.
    - hyperparameters: Dictionary of the best hyperparameters.
    - metrics: Dictionary of the model's performance metrics.
    - folder: The folder where the files should be saved.
    """
    os.makedirs(folder, exist_ok=True)
    model_path = os.path.join(folder, "model.joblib")
    hyperparameters_path = os.path.join(folder, "hyperparameters.json")
    metrics_path = os.path.join(folder, "metrics.json")

    dump(model, model_path)

    with open(hyperparameters_path, 'w') as hp_file:
        json.dump(hyperparameters, hp_file, indent=4)

    with open(metrics_path, 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    print(f"Model, hyperparameters, and metrics saved to {folder}")

def tune_classification_model_hyperparameters(model_class, X_train, y_train, X_test, y_test, param_grid):
    """
    Tunes the hyperparameters of a classification model using GridSearchCV.
    """
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_class())
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    test_predictions = best_model.predict(X_test)
    performance_metrics = {
        'test_accuracy': accuracy_score(y_test, test_predictions),
        'test_precision': precision_score(y_test, test_predictions, average='weighted'),
        'test_recall': recall_score(y_test, test_predictions, average='weighted'),
        'test_f1': f1_score(y_test, test_predictions, average='weighted')
    }

    return best_model, best_hyperparameters, performance_metrics

def tune_and_save_decision_tree(X_train, y_train, X_test, y_test):
    decision_tree_param_grid = {
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    best_model, best_hyperparameters, performance_metrics = tune_classification_model_hyperparameters(
        DecisionTreeClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        decision_tree_param_grid
    )
    save_model(best_model, best_hyperparameters, performance_metrics, folder="models/classification/decision_tree")

def tune_and_save_random_forest(X_train, y_train, X_test, y_test):
    random_forest_param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    best_model, best_hyperparameters, performance_metrics = tune_classification_model_hyperparameters(
        RandomForestClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        random_forest_param_grid
    )
    save_model(best_model, best_hyperparameters, performance_metrics, folder="models/classification/random_forest")

def tune_and_save_gradient_boosting(X_train, y_train, X_test, y_test):
    gradient_boosting_param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 4, 5]
    }
    best_model, best_hyperparameters, performance_metrics = tune_classification_model_hyperparameters(
        GradientBoostingClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        gradient_boosting_param_grid
    )
    save_model(best_model, best_hyperparameters, performance_metrics, folder="models/classification/gradient_boosting")

def evaluate_all_models(task_folder):
    models = [
        ("decision_tree", "models/classification/decision_tree"),
        ("random_forest", "models/classification/random_forest"),
        ("gradient_boosting", "models/classification/gradient_boosting")
    ]

    for model_name, folder in models:
        model_path = os.path.join(folder, "model.joblib")
        hyperparameters_path = os.path.join(folder, "hyperparameters.json")
        metrics_path = os.path.join(folder, "metrics.json")

        model = load(model_path)

        with open(hyperparameters_path, 'r') as hp_file:
            hyperparameters = json.load(hp_file)
        with open(metrics_path, 'r') as metrics_file:
            metrics = json.load(metrics_file)

        print(f"Model: {model_name}")
        print(f"Hyperparameters: {hyperparameters}")
        print(f"Performance Metrics: {metrics}")
        print("\n")

def find_best_model(task_folder):
    
    model_folders = [d for d in os.listdir(task_folder) if os.path.isdir(os.path.join(task_folder, d))]
    
    best_model = None
    best_metrics = None
    best_hyperparameters = None
    best_model_name = None

    for model_name in model_folders:
        folder = os.path.join(task_folder, model_name)
        model_path = os.path.join(folder, "model.joblib")
        metrics_path = os.path.join(folder, "metrics.json")

        if not os.path.exists(model_path) or not os.path.exists(metrics_path):
            continue

        model = load(model_path)
        with open(metrics_path, 'r') as metrics_file:
            metrics = json.load(metrics_file)

        # Compare metrics and choose the best one
        if best_metrics is None or metrics['test_accuracy'] > best_metrics['test_accuracy']:
            best_model = model
            best_metrics = metrics
            best_hyperparameters_path = os.path.join(folder, "hyperparameters.json")
            with open(best_hyperparameters_path, 'r') as hp_file:
                best_hyperparameters = json.load(hp_file)
            best_model_name = model_name

    return best_model, best_hyperparameters, best_metrics

if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('clean_tabular_data.csv')

    # Encode categorical labels
    label_encoder = LabelEncoder()
    data['Category'] = label_encoder.fit_transform(data['Category'])

    # Load the features and labels
    def load_airbnb(data, label):
        features = data.drop(columns=[label])
        labels = data[label]
        return features, labels

    X, y = load_airbnb(data, label='Category')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tune and save models
    tune_and_save_decision_tree(X_train, y_train, X_test, y_test)
    tune_and_save_random_forest(X_train, y_train, X_test, y_test)
    tune_and_save_gradient_boosting(X_train, y_train, X_test, y_test)

    # Evaluate all models
    evaluate_all_models(task_folder="models/classification")

    # Find and print the best model
    best_model, best_hyperparameters, best_metrics = find_best_model(task_folder="models/classification")
    print("Best Model:")
    print(f"Hyperparameters: {best_hyperparameters}")
    print(f"Performance Metrics: {best_metrics}")
