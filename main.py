"""
    using a decision tree, random forest, support vector machine,
    and k-nearest neighbors to predict the target variable
    based on the features in the dataset.

    The code includes:
    - data handling
    - model training
    - evaluation
    - visualization
"""

# =========================
# DATA HANDLING
# =========================
import pandas as pd
import numpy as np

# Train/test split
from sklearn.model_selection import train_test_split

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix

# Scaling
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt
from sklearn import tree


# =========================
# LOAD DATASET
# =========================
data = pd.read_csv('data.csv')


# =========================
# SHOW DATASET INFO
# =========================
def show_dataset_info(data):

    print("=== COLUMN NAMES ===")
    print(data.columns)

    print("\n=== FIRST 5 ROWS ===")
    print(data.head())

    print("\n=== DATA TYPES ===")
    print(data.dtypes)

    print("\n=== SIZE (ROWS, COLUMNS) ===")
    print(data.shape)


# =========================
# CLEAN DATA
# =========================
def clean_data(data):

    # Remove unnecessary columns
    data = data.drop(columns=["id", "Unnamed: 32"])

    # Convert diagnosis labels
    data["diagnosis"] = data["diagnosis"].map({
        "M": 1,
        "B": 0
    })

    return data


# =========================
# SPLIT DATA
# =========================
def split_data(data):

    # Features
    X = data.drop(columns=["diagnosis"])

    # Target
    y = data["diagnosis"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# =========================
# SCALE DATA
# =========================
def scale_data(X_train, X_test):

    scaler = StandardScaler()

    # Fit and transform training data
    X_train = scaler.fit_transform(X_train)

    # Transform testing data
    X_test = scaler.transform(X_test)

    return X_train, X_test


# =========================
# DECISION TREE
# =========================
def train_decision_tree(X_train, X_test, y_train, y_test):

    # Create model
    tree_model = DecisionTreeClassifier(random_state=42)

    # Train model
    tree_model.fit(X_train, y_train)

    # Predictions
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


# =========================
# VISUALIZE DECISION TREE
# =========================
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


# =========================
# RANDOM FOREST MAP REDUCE
# =========================
def train_random_forest_map_reduce(X_train, X_test, y_train, y_test):

    depths = [3, 5, 7]

    trees = []

    predictions = []

    # MAP STEP
    for depth in depths:

        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        trees.append(model)

        predictions.append(y_pred)

    # REDUCE STEP
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


# =========================
# VISUALIZE RANDOM FOREST
# =========================
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


# =========================
# VISUALIZE SVM BOUNDARY
# =========================
def visualize_svm_boundary(model, X, y, title):

    # Mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    # Decision function
    Z = model.decision_function(
        np.c_[xx.ravel(), yy.ravel()]
    )

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))

    # Boundary + margins
    plt.contour(
        xx,
        yy,
        Z,
        levels=[-1, 0, 1],
        linestyles=['--', '-', '--']
    )

    # Background regions
    plt.contourf(
        xx,
        yy,
        Z > 0,
        alpha=0.3
    )

    # Data points
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors='k'
    )

    # Support vectors
    plt.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=120,
        facecolors='none',
        edgecolors='red',
        linewidths=2,
        label='Support Vectors'
    )

    plt.xlabel("radius_mean")

    plt.ylabel("texture_mean")

    plt.title(title)

    plt.legend()

    plt.show()


# =========================
# SUPPORT VECTOR MACHINES
# =========================
def train_svm_models(cleaned_data):

    # Use only 2 features for visualization
    X = cleaned_data[["radius_mean", "texture_mean"]]

    # Target
    y = cleaned_data["diagnosis"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scale data
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    # =========================
    # RBF KERNEL
    # =========================
    rbf_model = SVC(
        kernel='rbf',
        C=1,
        gamma='scale',
        random_state=42
    )

    rbf_model.fit(X_train, y_train)

    rbf_predictions = rbf_model.predict(X_test)

    rbf_cm = confusion_matrix(
        y_test,
        rbf_predictions
    )

    rbf_accuracy = accuracy_score(
        y_test,
        rbf_predictions
    )

    print("\n=== SVM RBF RESULTS ===")

    print("Confusion Matrix:")
    print(rbf_cm)

    print("Accuracy:", rbf_accuracy)

    visualize_svm_boundary(
        rbf_model,
        X_train,
        y_train,
        "SVM Decision Boundary - RBF Kernel"
    )

    # =========================
    # POLYNOMIAL KERNEL
    # =========================
    poly_model = SVC(
        kernel='poly',
        degree=3,
        C=1,
        gamma='scale',
        random_state=42
    )

    poly_model.fit(X_train, y_train)

    poly_predictions = poly_model.predict(X_test)

    poly_cm = confusion_matrix(
        y_test,
        poly_predictions
    )

    poly_accuracy = accuracy_score(
        y_test,
        poly_predictions
    )

    print("\n=== SVM POLYNOMIAL RESULTS ===")

    print("Confusion Matrix:")
    print(poly_cm)

    print("Accuracy:", poly_accuracy)

    visualize_svm_boundary(
        poly_model,
        X_train,
        y_train,
        "SVM Decision Boundary - Polynomial Kernel"
    )


# =========================
# K-NEAREST NEIGHBORS
# =========================
def train_knn_models(cleaned_data):

    # Use same 2 features
    X = cleaned_data[["radius_mean", "texture_mean"]]

    # Target
    y = cleaned_data["diagnosis"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scale data
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    # Store results
    results = []

    # Distance metrics
    metrics = {
        "Euclidean": "euclidean",
        "Manhattan": "manhattan"
    }

    # K values
    k_values = [3, 5]

    # Test all combinations
    for metric_name, metric_value in metrics.items():

        for k in k_values:

            print("\n==============================")
            print(f"KNN USING {metric_name} DISTANCE | k = {k}")
            print("==============================")

            # Create model
            knn_model = KNeighborsClassifier(
                n_neighbors=k,
                metric=metric_value
            )

            # Train model
            knn_model.fit(X_train, y_train)

            # Predictions
            predictions = knn_model.predict(X_test)

            # Confusion matrix
            cm = confusion_matrix(y_test, predictions)

            # TN, FP, FN, TP
            tn, fp, fn, tp = cm.ravel()

            # Metrics
            accuracy = accuracy_score(y_test, predictions)

            sensitivity = tp / (tp + fn)

            specificity = tn / (tn + fp)

            # Print results
            print("Confusion Matrix:")
            print(cm)

            print("Accuracy:", accuracy)

            print("Sensitivity:", sensitivity)

            print("Specificity:", specificity)

            # Save results
            results.append({
                "Distance": metric_name,
                "K": k,
                "Accuracy": accuracy,
                "Sensitivity": sensitivity,
                "Specificity": specificity
            })

    # =========================
    # COMPARISON RESULTS
    # =========================
    print("\n========================================")
    print("KNN MODEL COMPARISON")
    print("========================================")

    for result in results:

        print(
            f"Distance: {result['Distance']} | "
            f"k = {result['K']} | "
            f"Accuracy = {result['Accuracy']:.4f} | "
            f"Sensitivity = {result['Sensitivity']:.4f} | "
            f"Specificity = {result['Specificity']:.4f}"
        )


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # Clean dataset
    cleaned_data = clean_data(data)

    print("\n=== CLEANED DATA INFO ===")

    show_dataset_info(cleaned_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(cleaned_data)

    # Scale data
    X_train, X_test = scale_data(
        X_train,
        X_test
    )

    # =========================
    # DECISION TREE
    # =========================
    tree_model = train_decision_tree(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Visualize decision tree
    visualize_decision_tree(
        tree_model,
        cleaned_data.drop(columns=["diagnosis"]).columns
    )

    # =========================
    # RANDOM FOREST
    # =========================
    forest_trees = train_random_forest_map_reduce(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Visualize random forest trees
    visualize_random_forest_trees(
        forest_trees,
        cleaned_data.drop(columns=["diagnosis"]).columns
    )

    # =========================
    # SUPPORT VECTOR MACHINE
    # =========================
    train_svm_models(cleaned_data)

    # =========================
    # K-NEAREST NEIGHBORS
    # =========================
    train_knn_models(cleaned_data)