import csv
import math
import random
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Making the output same in every run
random.seed(42)

# Dataset: Breast Cancer Dataset
file_path = "Assignment 1\Datasets\cancer.data"
X_raw = []
y_raw = []

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        y_raw.append(row[0])
        X_raw.append(row[1:])

enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = enc.fit_transform(X_raw)

label_mapping = {'no-recurrence-events': 0, 'recurrence-events': 1}
y = [label_mapping[label] for label in y_raw]

scaler = StandardScaler()
X = scaler.fit_transform(X_encoded)

# Distance Functions
def euclidean_distance(x1, x2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

def manhattan_distance(x1, x2):
    return sum(abs(a - b) for a, b in zip(x1, x2))

def minkowski_distance(x1, x2, p=3):
    return sum(abs(a - b) ** p for a, b in zip(x1, x2)) ** (1/p)

# Custom KNN Classifier
class KNN:
    def __init__(self, k=3, distance='euclidean', p=3):
        self.k = k
        self.distance = distance
        self.p = p

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        if self.distance == 'euclidean':
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance == 'manhattan':
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
        elif self.distance == 'minkowski':
            distances = [minkowski_distance(x, x_train, self.p) for x_train in self.X_train]

        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return most_common_label

# Accuracy calculation
def accuracy_score(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

# K-fold cross-validation
def k_fold_cross_validation(custom_model, sklearn_model, X, y, k=10):
    fold_size = len(X) // k
    custom_scores = []
    sklearn_scores = []
    indices = list(range(len(X)))
    random.shuffle(indices)

    for fold in range(k):
        start, end = fold * fold_size, (fold + 1) * fold_size if fold < k - 1 else len(X)
        test_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]

        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]

        custom_model.fit(X_train, y_train)
        custom_predictions = custom_model.predict(X_test)
        custom_score = accuracy_score(y_test, custom_predictions)
        custom_scores.append(custom_score)

        sklearn_model.fit(X_train, y_train)
        sklearn_predictions = sklearn_model.predict(X_test)
        sklearn_score = accuracy_score(y_test, sklearn_predictions)
        sklearn_scores.append(sklearn_score)

    return custom_scores, sklearn_scores

# Initialize and test custom KNN
knn_euclidean = KNN(k=3, distance='euclidean')
knn_manhattan = KNN(k=3, distance='manhattan')
knn_minkowski = KNN(k=3, distance='minkowski', p=3)

# Scikit-learn KNN
knn1 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn2 = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn3 = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=3)

# Perform K-fold cross-validation and collect accuracies
custom_scores_euclidean, sklearn_scores_euclidean = k_fold_cross_validation(knn_euclidean, knn1, X, y, k=10)
custom_scores_manhattan, sklearn_scores_manhattan = k_fold_cross_validation(knn_manhattan, knn2, X, y, k=10)
custom_scores_minkowski, sklearn_scores_minkowski = k_fold_cross_validation(knn_minkowski, knn3, X, y, k=10)

# Print accuracies
print(f"\nKNN Mean Accuracies: Custom KNN\n")
print(f"Using Euclidean distance: {sum(custom_scores_euclidean) / len(custom_scores_euclidean) * 100}%")
print(f"Using Manhattan distance: {sum(custom_scores_manhattan) / len(custom_scores_manhattan) * 100}%")
print(f"Using Minkowski distance: {sum(custom_scores_minkowski) / len(custom_scores_minkowski) * 100}%")

print(f"\nKNN Mean Accuracies: SciKit KNN\n")
print(f"Scikit-learn KNN Euclidean Mean Accuracy: {sum(sklearn_scores_euclidean) / len(sklearn_scores_euclidean) * 100}%")
print(f"Scikit-learn KNN Manhattan Mean Accuracy: {sum(sklearn_scores_manhattan) / len(sklearn_scores_manhattan) * 100}%")
print(f"Scikit-learn KNN Minkowski Mean Accuracy: {sum(sklearn_scores_minkowski) / len(sklearn_scores_minkowski) * 100}%")

def custom_paired_t_test(sample1, sample2):
    differences = [a - b for a, b in zip(sample1, sample2)]
    mean_diff = sum(differences) / len(differences)
    variance = sum((d - mean_diff) ** 2 for d in differences) / (len(differences) - 1)
    std_diff = math.sqrt(variance)
    n = len(differences)
    if std_diff == 0:
        if mean_diff == 0:
            return 0, n - 1
        else:
            return float('inf') if mean_diff > 0 else float('-inf'), n - 1
    
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    df = n - 1

    return t_stat, df


t_stat_euclidean, df_euclidean = custom_paired_t_test(custom_scores_euclidean, sklearn_scores_euclidean)
t_stat_manhattan, df_manhattan = custom_paired_t_test(custom_scores_manhattan, sklearn_scores_manhattan)
t_stat_minkowski, df_minkowski = custom_paired_t_test(custom_scores_minkowski, sklearn_scores_minkowski)

alpha = 0.05
print(f"\nPaired T-test results for KNN using Euclidean distance:")
print(f"T-statistic: {t_stat_euclidean}, Degrees of Freedom: {df_euclidean}")
if abs(t_stat_euclidean) > 1.96:
    print("Significant difference between custom and Scikit-learn KNN (Euclidean). We accept the Null hypothesis.")
else:
    print("No significant difference between custom and Scikit-learn KNN (Euclidean). We reject the Null hypothesis.")

print(f"\nPaired T-test results for KNN using Manhattan distance:")
print(f"T-statistic: {t_stat_manhattan}, Degrees of Freedom: {df_manhattan}")
if abs(t_stat_manhattan) > 1.96:
    print("Significant difference between custom and Scikit-learn KNN (Manhattan). We accept the Null hypothesis.")
else:
    print("No significant difference between custom and Scikit-learn KNN (Manhattan). We reject the Null hypothesis.")

print(f"\nPaired T-test results for KNN using Minkowski distance:")
print(f"T-statistic: {t_stat_minkowski}, Degrees of Freedom: {df_minkowski}")
if abs(t_stat_minkowski) > 1.96:
    print("Significant difference between custom and Scikit-learn KNN (Minkowski). We accept the Null hypothesis.")
else:
    print("No significant difference between custom and Scikit-learn KNN (Minkowski). We reject the Null hypothesis.")
