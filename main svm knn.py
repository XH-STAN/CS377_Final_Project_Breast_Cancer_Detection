"""
Using Decision Tree, Random Forest, Support Vector Machine (SVM),
and K-Nearest Neighbors (K-NN) classifiers to predict the target
variable based on the features in the dataset.

Includes:
- Data handling
- Model training
- Performance evaluation
- Visualization
- K-NN comparison using:
    * Manhattan Distance
    * Euclidean Distance
    * k = 3 and k = 5
"""

# Data handling
import pandas as pd
import numpy as np

# Train/test split
from sklearn.model_selection import train_test_split

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Visualization
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the dataset
data = pd.read_csv('data.csv')


def print_data_info(data):
    """Prints basic information and statistics about the dataset."""
    print("Data Info:")
    print(data.info())

    print("\nData Description:")
    print(data.describe())


# ==========================
# Feature Selection
# ==========================
# Using two features for visualization
X = data[["radius_mean", "texture_mean"]]

# Target column
if data["diagnosis"].dtype == object:
    y = data["diagnosis"].map({'M': 1, 'B': 0})
else:
    y = data["diagnosis"]


# ==========================
# Train/Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================
# SVM Models
# ==========================
svm_rbf = SVC(kernel='rbf', gamma='scale', C=1.0)

svm_poly = SVC(
    kernel='poly',
    degree=3,
    gamma='scale',
    C=1.0
)

# Train SVM models
svm_rbf.fit(X_train, y_train)
svm_poly.fit(X_train, y_train)

# Predictions
rbf_pred = svm_rbf.predict(X_test)
poly_pred = svm_poly.predict(X_test)


# ==========================
# K-NN Models
# ==========================

# Euclidean Distance = minkowski with p=2
knn_euclidean_k3 = KNeighborsClassifier(
    n_neighbors=3,
    metric='minkowski',
    p=2
)

knn_euclidean_k5 = KNeighborsClassifier(
    n_neighbors=5,
    metric='minkowski',
    p=2
)

# Manhattan Distance = minkowski with p=1
knn_manhattan_k3 = KNeighborsClassifier(
    n_neighbors=3,
    metric='minkowski',
    p=1
)

knn_manhattan_k5 = KNeighborsClassifier(
    n_neighbors=5,
    metric='minkowski',
    p=1
)

# Train K-NN models
knn_euclidean_k3.fit(X_train, y_train)
knn_euclidean_k5.fit(X_train, y_train)

knn_manhattan_k3.fit(X_train, y_train)
knn_manhattan_k5.fit(X_train, y_train)

# Predictions
euclidean_k3_pred = knn_euclidean_k3.predict(X_test)
euclidean_k5_pred = knn_euclidean_k5.predict(X_test)

manhattan_k3_pred = knn_manhattan_k3.predict(X_test)
manhattan_k5_pred = knn_manhattan_k5.predict(X_test)


# ==========================
# Evaluation Function
# ==========================
def evaluate_model(name, y_test, y_pred):

    print(f"\n===== {name} =====")

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy


# ==========================
# Evaluate SVM Models
# ==========================
rbf_accuracy = evaluate_model(
    "SVM - RBF Kernel",
    y_test,
    rbf_pred
)

poly_accuracy = evaluate_model(
    "SVM - Polynomial Kernel",
    y_test,
    poly_pred
)


# ==========================
# Evaluate K-NN Models
# ==========================
euclidean_k3_accuracy = evaluate_model(
    "K-NN Euclidean (k=3)",
    y_test,
    euclidean_k3_pred
)

euclidean_k5_accuracy = evaluate_model(
    "K-NN Euclidean (k=5)",
    y_test,
    euclidean_k5_pred
)

manhattan_k3_accuracy = evaluate_model(
    "K-NN Manhattan (k=3)",
    y_test,
    manhattan_k3_pred
)

manhattan_k5_accuracy = evaluate_model(
    "K-NN Manhattan (k=5)",
    y_test,
    manhattan_k5_pred
)


# ==========================
# Compare Results
# ==========================
print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(f"SVM RBF Accuracy: {rbf_accuracy:.4f}")
print(f"SVM Polynomial Accuracy: {poly_accuracy:.4f}")

print(f"K-NN Euclidean k=3 Accuracy: {euclidean_k3_accuracy:.4f}")
print(f"K-NN Euclidean k=5 Accuracy: {euclidean_k5_accuracy:.4f}")

print(f"K-NN Manhattan k=3 Accuracy: {manhattan_k3_accuracy:.4f}")
print(f"K-NN Manhattan k=5 Accuracy: {manhattan_k5_accuracy:.4f}")


# ==========================
# Decision Boundary Plot
# ==========================
def plot_decision_boundary(model, X, y, title):

    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1

    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1)
    )

    Z = model.predict(
        np.c_[xx.ravel(), yy.ravel()]
    )

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    plt.contourf(
        xx,
        yy,
        Z,
        alpha=0.3,
        cmap=ListedColormap(
            ('lightblue', 'lightcoral')
        )
    )

    plt.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=y,
        edgecolor='k',
        cmap=ListedColormap(('blue', 'red'))
    )

    plt.xlabel("Radius Mean")
    plt.ylabel("Texture Mean")
    plt.title(title)

    plt.show()


# ==========================
# Plot K-NN Decision Boundaries
# ==========================
plot_decision_boundary(
    knn_euclidean_k3,
    X_test,
    y_test,
    "K-NN Euclidean Distance (k=3)"
)

plot_decision_boundary(
    knn_euclidean_k5,
    X_test,
    y_test,
    "K-NN Euclidean Distance (k=5)"
)

plot_decision_boundary(
    knn_manhattan_k3,
    X_test,
    y_test,
    "K-NN Manhattan Distance (k=3)"
)

plot_decision_boundary(
    knn_manhattan_k5,
    X_test,
    y_test,
    "K-NN Manhattan Distance (k=5)"
)


# ==========================
# Original Scatter Plot
# ==========================
def graph(df):

    fig = plt.figure()

    plt.scatter(
        df["radius_mean"],
        df["texture_mean"]
    )

    plt.xlabel("Radius Mean")
    plt.ylabel("Texture Mean")
    plt.title("Radius vs Texture")

    plt.show()

    return not plt.fignum_exists(fig.number)


if __name__ == "__main__":

    try:
        print("Running Classification Models...")
        print_data_info(data)

        closed = graph(data)

        if closed:
            print("Visualization closed.")

    except KeyboardInterrupt:
        print("\nStopped by user")