## Modelling Airbnb's Property Listing Data Analysis Project

## Project Overview

This project aims to build, train, and evaluate various machine learning models to predict Airbnb nightly prices (regression task) and to classify listings into categories (classification task). The project includes data preprocessing, model training, hyperparameter tuning, and evaluation for both regression and classification models. The best-performing models are selected based on validation set performance.

## Table of Contents

Project Overview
Dataset
Models
 - Neural Network Model
 - Regression Models
 - Classification Models
Training and Evaluation
Hyperparameter Tuning
Installation
Usage
Project Structure
Results

## Dataset

The project utilizes a dataset of Airbnb listings. The features include various attributes of the listings like the number of bedrooms, location, amenities, etc. The target variables include:

Night_Price : Used for regression tasks.
Category : Used for classification tasks.

## Prerequisites

Ensure the dataset (AirBnbData.csv) is present in the root directory of the project. Specify the target columns for the tasks (e.g., price for regression and category for classification) in the scripts.

## Models

# Neural Network Model
The neural network is a fully connected model built with PyTorch, designed for both regression and classification tasks. The architecture is dynamic, allowing customization through a configuration file (nn_config.yaml). The model can output either a single value for regression tasks or a set of class probabilities for classification tasks.

# Configuration

The model architecture, learning rate, optimizer, and other hyperparameters are defined in a YAML file nn_config.yaml: 

optimiser: Adam
learning_rate: 0.001
hidden_layer_width: 256
depth: 3


## Regression Models
In addition to the neural network, the project also includes traditional regression models, such as:

- Linear Regression
- Random Forest Regression
- Support Vector Regression
These models are implemented using Scikit-learn and are evaluated using RMSE and R-squared metrics.

## Classification Models
For the classification task, the project includes several classification models, such as:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
These models are also implemented using Scikit-learn and are evaluated using accuracy, precision, recall, and F1-score.

## Training and Evaluation

# Regression Tasks
For regression tasks, the script trains and evaluates models based on RMSE and R-squared metrics. The best model is selected based on validation performance.

# Classification Tasks
For classification tasks, the script evaluates models using accuracy, precision, recall, and F1-score. The best model is selected based on these metrics.

# Gradient Clipping
For neural networks, gradient clipping is applied during training to prevent exploding gradients.

# Hyperparameter Tuning

The project includes a hyperparameter search for both regression and classification tasks. Multiple configurations are tested, and the best model is selected based on validation performance.

## Installation

# Requirements
  - Python 3.7+
  - PyTorch
  - Scikit-learn
  - Pandas
  - NumPy
  - PyYAML

## Steps

1. Clone the repository:
  - git clone https://github.com/your-username/modelling-airbnbs-property-listing-dataset

2. Install the required packages:
  - pip install -r requirements.txt

3. Place your dataset (AirBnbData.csv) in the project directory.

## Usage

 -  Training the Models
 -  To train models and find the best configuration:
    - Preprocess the data.
    - Train multiple models with different configurations.
    - Save the best models and their metrics.

# Configuration
Modify the nn_config.yaml file to adjust the model architecture and training settings for the neural network. For traditional models, you can adjust parameters directly in the scripts or create a similar configuration file.

# Running the Scripts
Run the appropriate script based on your task

## Project Structure

airbnb-models/
│
├── models/                           # Directory for saving trained models
│   ├── neural_networks/              # Directory for neural network models
│   │   ├── regression/
│   │   │   └── best_model/           # Best regression model for neural networks
│   │   └── classification/
│   │       └── best_model/           # Best classification model for neural networks
│   ├── regression/                   # Directory for traditional regression models
│   └── classification/               # Directory for traditional classification models
│
├── nn_config.yaml                    # YAML configuration file for neural networks
├── AirBnbData.csv                    # Dataset (not included in the repository)
├── requirements.txt                  # Python package dependencies
├── train_regression.py               # Script for training regression models
├── train_classification.py           # Script for training classification models
└── README.md                         # Project README file

## Results

After running the scripts, the best-performing models and their configurations will be saved in the respective directories under models folder. The results include metrics such as RMSE, R-squared for regression models, and accuracy, precision, recall, and F1-score for classification models.
    