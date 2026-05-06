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







def graph(df):
    """Creates a scatter plot of radius_mean vs texture_mean and waits until the plot is closed."""
    fig = plt.figure()
    plt.scatter(df["radius_mean"], df["texture_mean"])
    plt.xlabel("Radius Mean")
    plt.ylabel("Texture Mean")
    plt.title("Radius vs Texture")

    plt.show()  # waits until closed

    return not plt.fignum_exists(fig.number)

if __name__ == "__main__":
    try:
        while True:
            print("Running...")
            print_data_info(data)
            
            closed = graph(data)
            if closed:
                break

    except KeyboardInterrupt:
        print("\nStopped by user")