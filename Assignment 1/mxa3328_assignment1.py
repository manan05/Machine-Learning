# Assignment 1
# 1002143328

# imports

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Euclidean Distance
def euclidean_distance(x1, x2):
    distance_squared = np.sum((x1-x2) ** 2)
    distance = np.sqrt(distance_squared)
    return distance

# Calculate KNN Function
def knn(X_train, y_train, X_test, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_test, X_train[i])
        distances.append((dist, y_train[i]))
    
    k_neighbors = sorted(distances, key=lambda x: x[0])[:k]
    k_nearest_labels = [label for _, label in k_neighbors]
    most_common = Counter(k_nearest_labels).most_common(1)
    
    return most_common[0][0]

# k-Fold Cross Validation
def k_fold_cross_validation(X, y, k_neighbors, k_folds=10):
    fold_size = len(X) // k_folds
    accuracies = []
    
    for i in range(k_folds):
        start, end = i * fold_size, (i + 1) * fold_size
        X_val, y_val = X[start:end], y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        
        correct = 0
        for j in range(len(X_val)):
            prediction = knn(X_train, y_train, X_val[j], k_neighbors)
            if prediction == y_val[j]:
                correct += 1
                
        accuracy = correct / len(X_val)
        accuracies.append(accuracy)
    
    return np.mean(accuracies)

# Comparison with Scikit-Learn KNN
def compare_with_sklearn(X, y, k_neighbors):
    knn_sklearn = KNeighborsClassifier(n_neighbors=k_neighbors)
    scores = cross_val_score(knn_sklearn, X, y, cv=10)
    return np.mean(scores)

def main():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Set parameters
    k_neighbors = 5  # Choose the number of neighbors
    
    # Evaluate your KNN implementation with 10-fold cross-validation
    accuracy_knn = k_fold_cross_validation(X, y, k_neighbors)
    print(f'My KNN accuracy: {accuracy_knn}')
    
    # Evaluate Scikit-Learn KNN with 10-fold cross-validation
    accuracy_sklearn = compare_with_sklearn(X, y, k_neighbors)
    print(f'Scikit-Learn KNN accuracy: {accuracy_sklearn}')
    
    # You can also perform hypothesis testing here for a statistical comparison

if __name__ == '__main__':
    main()