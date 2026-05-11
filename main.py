"""
    using a decision tree and random forest to predict the target variable based on the features in the dataset. The code includes data handling, model training, evaluation, and visualization of results.

"""

#data handling
import pandas as pd
import numpy as np

# Train/test split
from sklearn.model_selection import train_test_split

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix

# Visualization
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data.csv')






def show_dataset_info(data):
    """Prints basic information about the dataset, including column names, data types, missing values, and shape."""
    print("=== COLUMN NAMES ===")
    print(data.columns)

    print("\n=== FIRST 5 ROWS ===")
    print(data.head())

    print("\n=== DATA TYPES ===")
    print(data.dtypes)

    print("\n=== size (rows, columns) ===")
    print(data.shape)


def clean_data(data):
    """remove all the useless columns and convert the diagnosis labels to binary values (1 for malignant and 0 for benign) and returns the cleaned dataset."""
    data = data.drop(columns=["id", "Unnamed: 32"])

    # Convert diagnosis labels
    data["diagnosis"] = data["diagnosis"].map({
        "M": 1,
        "B": 0
    })

    return data




def split_data(data):
    """Splits the cleaned dataset into training and testing sets, with 80% of the data used for training and 20% for testing. Returns the feature columns (X) and target column (y) for both training and testing sets."""
    # X = feature columns
    X = data.drop(columns=["diagnosis"])

    # y = target column
    y = data["diagnosis"]

    # Split into training/testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test







if __name__ == "__main__":



    #clear the data and print out the cleaned dataset information to verify that the cleaning process has been successful and that the dataset is ready for modeling.
    cleaned_data = clean_data(data)

    print("\n=== CLEANED DATA INFO ===")
    show_dataset_info(cleaned_data)



 #print the data out after the split to verify the shapes of the training and testing sets. This will help ensure that the data has been split correctly and that the feature and target variables are properly separated.
    X_train, X_test, y_train, y_test = split_data(cleaned_data)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)