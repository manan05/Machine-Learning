import csv
import math
import random
from sklearn.neighbors import KNeighborsClassifier

# Dataset: Car Evaluation Dataset
file_path = "Machine-Learning/Assignment 1/Datasets/car.data"
X = []
y = []

label_mapping = {}
feature_labels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
mapping_count = 0

# Load and preprocess the dataset
with open(file_path, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        features = row[:-1]
        numerical_features = []
        for i, feature in enumerate(features):
            if feature not in label_mapping:
                label_mapping[feature] = mapping_count
                mapping_count += 1
            numerical_features.append(label_mapping[feature])
        X.append(numerical_features)
        
        label = row[-1]
        if label not in label_mapping:
            label_mapping[label] = mapping_count
            mapping_count += 1
        y.append(label_mapping[label])

print("Feature Mapping:")
for feature, index in label_mapping.items():
    print(f"{feature}: {index}")

print("\nFirst 5 Transformed Features (X):")
for i in range(5):
    print(X[i])

print("\nFirst 5 Labels (y):")
for i in range(5):
    print(y[i])

def euclidean_distance(x1, x2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

def manhattan_distance(x1, x2):
    return sum(abs(a - b) for a, b in zip(x1, x2))

def minkowski_distance(x1, x2, p=3):
    return sum(abs(a - b) ** p for a, b in zip(x1, x2)) ** (1 / p)

# Custom KNN Classifier
class KNN:
    def __init__(self, k=5, distance="euclidean", p=3):
        self.k = k
        self.distance = distance
        self.p = p

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        if self.distance == "euclidean":
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance == "manhattan":
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
        elif self.distance == "minkowski":
            distances = [minkowski_distance(x, x_train, self.p) for x_train in self.X_train]
        
        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        label_count = {}
        for label in k_nearest_labels:
            label_count[label] = label_count.get(label, 0) + 1

        return max(label_count, key=label_count.get)

# Accuracy calculation
def accuracy_score(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true) if y_true else 0

# Custom K-Fold Cross-Validation
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

# Custom KNN Models
knn_euclidean = KNN(k=5, distance="euclidean")
knn_manhattan = KNN(k=5, distance="manhattan")
knn_minkowski = KNN(k=5, distance="minkowski", p=3)

# Scikit-learn KNN
knn1 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn2 = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn3 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)

# Perform K-fold cross-validation
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

# Paired T-tests using custom t-test function
t_stat_euclidean, df_euclidean = custom_paired_t_test(custom_scores_euclidean, sklearn_scores_euclidean)
t_stat_manhattan, df_manhattan = custom_paired_t_test(custom_scores_manhattan, sklearn_scores_manhattan)
t_stat_minkowski, df_minkowski = custom_paired_t_test(custom_scores_minkowski, sklearn_scores_minkowski)

alpha = 0.05

# For a two-tailed test at alpha = 0.05, we compare t-statistic to critical value ~1.96 (normal distribution assumption)

print(f"\nCustom Paired T-test results for KNN using Euclidean distance:")
print(f"T-statistic: {t_stat_euclidean}, Degrees of Freedom: {df_euclidean}")
if abs(t_stat_euclidean) > 1.96:
    print("Significant difference between custom and Scikit-learn KNN (Euclidean).")
else:
    print("No significant difference between custom and Scikit-learn KNN (Euclidean).")

print(f"\nCustom Paired T-test results for KNN using Manhattan distance:")
print(f"T-statistic: {t_stat_manhattan}, Degrees of Freedom: {df_manhattan}")
if abs(t_stat_manhattan) > 1.96:
    print("Significant difference between custom and Scikit-learn KNN (Manhattan).")
else:
    print("No significant difference between custom and Scikit-learn KNN (Manhattan).")

print(f"\nCustom Paired T-test results for KNN using Minkowski distance:")
print(f"T-statistic: {t_stat_minkowski}, Degrees of Freedom: {df_minkowski}")
if abs(t_stat_minkowski) > 1.96:
    print("Significant difference between custom and Scikit-learn KNN (Minkowski).")
else:
    print("No significant difference between custom and Scikit-learn KNN (Minkowski).")
