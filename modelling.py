from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from tabular_data import load_airbnb
import pandas as pd


# Load the Airbnb dataset
data = pd.read_csv('clean_tabular_data.csv')  

# Extract features and labels
features, labels = load_airbnb(data, label='Price_Night')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Impute missing values in the features with the mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train a linear regression model using the imputed data
model = SGDRegressor(max_iter=10000, random_state=42)
model.fit(X_train_imputed, y_train)

# Make predictions on the test set
predictions = model.predict(X_test_imputed)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Compute RMSE and R^2 for training set
train_predictions = model.predict(X_train_imputed)
train_rmse = mean_squared_error(y_train, train_predictions, squared=False)  # RMSE
train_r2 = r2_score(y_train, train_predictions)  # R^2

# Compute RMSE and R^2 for test set
test_predictions = model.predict(X_test_imputed)
test_rmse = mean_squared_error(y_test, test_predictions, squared=False)  # RMSE
test_r2 = r2_score(y_test, test_predictions)  # R^2

print("Training Set:")
print("RMSE:", train_rmse)
print("R^2:", train_r2)
print("\nTest Set:")
print("RMSE:", test_rmse)
print("R^2:", test_r2)

