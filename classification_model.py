import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from tabular_data import load_airbnb  

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

    # Make predictions on the test set
    y_pred = model.predict(X_test_prepared)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Print a detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

