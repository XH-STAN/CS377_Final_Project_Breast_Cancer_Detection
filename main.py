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
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


# Evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix

# Visualization
import matplotlib.pyplot as plt
from sklearn import tree


# Load the dataset into memory 
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


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform it
    X_train = scaler.fit_transform(X_train)

    # Apply same scaling to test data
    X_test = scaler.transform(X_test)

    return X_train, X_test




def train_decision_tree(X_train, X_test, y_train, y_test):
    # Create decision tree model
    tree_model = DecisionTreeClassifier(random_state=42)

    # Train the model
    tree_model.fit(X_train, y_train)

    # Make predictions
    y_pred = tree_model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("\n=== DECISION TREE RESULTS ===")
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)

    return tree_model



def visualize_decision_tree(tree_model, feature_names):
    plt.figure(figsize=(20, 10))

    tree.plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=["Benign", "Malignant"],
        filled=True
    )

    plt.title("Decision Tree Visualization")
    plt.show()





def train_random_forest_map_reduce(X_train, X_test, y_train, y_test):
    depths = [3, 5, 7]
    trees = []
    predictions = []

    # MAP STEP: train 3 different trees
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        trees.append(model)
        predictions.append(y_pred)

    # REDUCE STEP: majority vote
    predictions = np.array(predictions)

    final_predictions = []

    for i in range(predictions.shape[1]):
        votes = predictions[:, i]
        majority_vote = np.bincount(votes).argmax()
        final_predictions.append(majority_vote)

    final_predictions = np.array(final_predictions)

    # Evaluation
    cm = confusion_matrix(y_test, final_predictions)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, final_predictions)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("\n=== RANDOM FOREST MAP-REDUCE RESULTS ===")
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)

    return trees



def visualize_random_forest_trees(trees, feature_names):
    depths = [3, 5, 7]

    for i, tree_model in enumerate(trees):
        plt.figure(figsize=(20, 10))

        tree.plot_tree(
            tree_model,
            feature_names=feature_names,
            class_names=["Benign", "Malignant"],
            filled=True
        )

        plt.title(f"Random Forest Tree (Depth = {depths[i]})")

        plt.show()





if __name__ == "__main__":



    #clear the data and print out the cleaned dataset information to verify that the cleaning process has been successful and that the dataset is ready for modeling.
    cleaned_data = clean_data(data)

    print("\n=== CLEANED DATA INFO ===")
    show_dataset_info(cleaned_data)

    #print the data out after the split to verify the shapes of the training and testing sets. This will help ensure that the data has been split correctly and that the feature and target variables are properly separated.
    X_train, X_test, y_train, y_test = split_data(cleaned_data)


    #scale the feature data to ensure that all features are on the same scale, which can improve the performance of many machine learning algorithms, including decision trees and random forests.
    X_train, X_test = scale_data(X_train, X_test)


    # train the decision tree model using the training data and evaluate its performance on the testing data. The function will print out the confusion matrix, accuracy, sensitivity, and specificity of the model, which are important metrics for assessing the performance of a classification model in the context of breast cancer detection.
    tree_model = train_decision_tree(X_train, X_test, y_train, y_test)
    
    
    # visualize the structure of the trained decision tree model using a plot. This visualization can help you understand how the model makes decisions based on the features in the dataset, and it can also provide insights into which features are most important for predicting breast cancer.
    visualize_decision_tree(tree_model, cleaned_data.drop(columns=["diagnosis"]).columns) 
    
    
    # train a random forest model using a map-reduce approach, where multiple decision trees are trained with different maximum depths (3, 5, and 7) and their predictions are combined using majority voting. The function will print out the confusion matrix, accuracy, sensitivity, and specificity of the random forest model, allowing you to compare its performance with the decision tree model.
    forest_trees = train_random_forest_map_reduce(X_train,X_test,y_train,y_test)
        
        
        
    # visualize the individual trees in the random forest model using plots. This can help you understand how each tree contributes to the overall predictions of the random forest and can provide insights into the importance of different features for predicting breast cancer.
    visualize_random_forest_trees(forest_trees,cleaned_data.drop(columns=["diagnosis"]).columns) 
