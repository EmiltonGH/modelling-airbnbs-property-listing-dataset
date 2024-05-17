import pandas as pd
import ast

def remove_rows_with_missing_ratings(dataset):
    """
    Remove rows with missing values in rating columns.
    
    Args:
    - dataset: pandas DataFrame containing the dataset
    
    Returns:
    - dataset: pandas DataFrame with rows containing missing values in rating columns removed
    """
    rating_columns = ['Cleanliness_rating', 'Accuracy_rating', 'Location_rating', 'Check-in_rating', 'Value_rating']
    dataset = dataset.dropna(subset=rating_columns)
    return dataset

def combine_description_strings(dataset):
    """
    Combine list items in the Description column into the same string.
    
    Args:
    - dataset: pandas DataFrame containing the dataset
    
    Returns:
    - dataset: pandas DataFrame with combined description strings
    """
    dataset = dataset.dropna(subset=['Description']) # Remove records with missing description
    # Check if the strings are in the form of lists
    mask = dataset['Description'].str.contains(r'\[', regex=False)
    dataset.loc[mask, 'Description'] = dataset.loc[mask, 'Description'].apply(lambda x: ' '.join(ast.literal_eval(x)))
    # Remove 'About this space' prefix and strip whitespace
    dataset.loc[:, 'Description'] = dataset['Description'].str.replace('About this space', '').str.strip()
    return dataset

def set_default_feature_values(dataset):
    """
    Replace empty values in 'guests', 'beds', 'bathrooms', and 'bedrooms' columns with 1.
    
    Args:
    - dataset: pandas DataFrame containing the dataset
    
    Returns:
    - dataset: pandas DataFrame with default feature values replaced
    """
    default_values = {'guests': 1, 'beds': 1, 'bathrooms': 1, 'bedrooms': 1}
    dataset.fillna(default_values, inplace=True)
    return dataset

def clean_tabular_data(raw_data):
    """
    Clean the raw tabular data.
    
    Args:
    - raw_data: pandas DataFrame containing the raw data
    
    Returns:
    - clean_data: pandas DataFrame containing the cleaned data
    """
    clean_data = raw_data.copy()
    clean_data = remove_rows_with_missing_ratings(clean_data)
    clean_data = combine_description_strings(clean_data)
    clean_data = set_default_feature_values(clean_data)
    return clean_data

def load_airbnb(data: pd.DataFrame, label: str):
    """
    Splits the DataFrame into features and labels.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the Airbnb data.
    label (str): The name of the column to be used as the label.

    Returns:
    tuple: A tuple containing the features DataFrame and the labels Series.
    """
    # Ensure the label column exists in the data
    if label not in data.columns:
        raise ValueError(f"Label column '{label}' does not exist in the data")

    # Filter out non-numerical columns
    numerical_data = data.select_dtypes(include=['number'])

    # Ensure the label column is in the numerical data
    if label not in numerical_data.columns:
        raise ValueError(f"Label column '{label}' is not numerical")

    # Extract the labels
    labels = numerical_data[label]

    # Remove the label column from the features
    features = numerical_data.drop(columns=[label])

    return features, labels

if __name__ == "__main__":
    # Load raw data
    raw_data = pd.read_csv('AirBnbData.csv')
    
    # Clean the data
    clean_data = clean_tabular_data(raw_data)
    
    # Save processed data
    clean_data.to_csv('clean_tabular_data.csv', index=False)

    # Use the function to get features and labels
    features, labels = load_airbnb(clean_data, label='Price_Night')

