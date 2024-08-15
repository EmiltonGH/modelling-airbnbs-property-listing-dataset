import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from tabular_data import load_airbnb  

def evaluate_classification_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model on training and test data, printing key performance metrics.
    """
    # Predictions on training data
    y_train_pred = model.predict(X_train)

    # Predictions on test data
    y_test_pred = model.predict(X_test)

    # Compute metrics for the training set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')

    # Compute metrics for the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Print the metrics
    print("Training Set Performance:")
    print(f"Accuracy: {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall: {train_recall:.4f}")
    print(f"F1 Score: {train_f1:.4f}")

    print("\nTest Set Performance:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")

if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('AirBnbData.csv')

    # Encode the 'Category' column
    label_encoder = LabelEncoder()
    data['Category'] = label_encoder.fit_transform(data['Category'])

    # Use load_airbnb to split features and labels
    features, labels = load_airbnb(data, label='Category')

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create a pipeline with an imputer and a scaler
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
        ('scaler', StandardScaler())  # Standardize features
    ])

    # Apply the pipeline to the training and test data
    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared = pipeline.transform(X_test)

    # Initialize and train the logistic regression model
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train_prepared, y_train)

    # Evaluate the model's performance
    evaluate_classification_model(model, X_train_prepared, y_train, X_test_prepared, y_test)
