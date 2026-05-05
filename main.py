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

def print_data_info(data):
    """Prints basic information and statistics about the dataset."""
    print("Data Info:")
    print(data.info())
    print("\nData Description:")
    print(data.describe())


if __name__ == "__main__":
    print_data_info(data)
