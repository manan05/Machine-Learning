import math
import random
import csv
from sklearn.neighbors import KNeighborsClassifier

# Making the output same in every run
random.seed(42)

# Dataset: Hayes-Roth Dataset
file_path = 'Assignment 1\Datasets\hayes-roth.data'
X = []
y = []

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        features = [float(row[i]) for i in range(len(row)) if i not in [1, 2]]
        label = int(row[2])
        X.append(features)
        y.append(label)

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
        else:
            raise ValueError("Unsupported distance metric")
        
        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        label_counts = {}
        for label in k_nearest_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        most_common_label = max(label_counts, key=label_counts.get)
        return most_common_label

# Accuracy Calculation
def accuracy_score(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

# Custom K-Fold Cross-Validation
def k_fold_cross_validation(model, X, y, k=10):
    fold_size = len(X) // k
    scores = []
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

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)
        scores.append(score)

    return sum(scores) / k, scores

# Custom Paired T-Test Function
def custom_ttest_rel(sample1, sample2):
    if len(sample1) != len(sample2):
        raise ValueError("Samples must have the same length")

    n = len(sample1)
    differences = [x1 - x2 for x1, x2 in zip(sample1, sample2)]
    mean_diff = sum(differences) / n
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in differences) / (n - 1))
    t_statistic = mean_diff / (std_diff / math.sqrt(n))

    # Calculate p-value for two-tailed test using the t-distribution
    p_value = 2 * (1 - cumulative_t_distribution(abs(t_statistic), n - 1))
    
    return t_statistic, p_value

def cumulative_t_distribution(t, df):
    # Implement the CDF for the t-distribution using numerical integration
    def integrand(x, df):
        numerator = (df + 1) ** 0.5
        denominator = math.sqrt(df * math.pi) * (1 + x ** 2) ** ((df + 1) / 2)
        return numerator / denominator
    
    a, b, n = -10, t, 10000  # Integration limits and number of slices
    h = (b - a) / n
    integral = 0.5 * (integrand(a, df) + integrand(b, df))  # Initial sum

    for i in range(1, n):
        x = a + i * h
        integral += integrand(x, df)

    integral *= h  # Final calculation of integral
    return integral

# Custom KNN Models
knn_euclidean = KNN(k=3, distance='euclidean')
knn_manhattan = KNN(k=3, distance='manhattan')
knn_minkowski = KNN(k=3, distance='minkowski', p=3) 

# Perform K-Fold Cross-Validation for Custom KNN
mean_accuracy1, accuracies1 = k_fold_cross_validation(knn_euclidean, X, y, k=10)
mean_accuracy2, accuracies2 = k_fold_cross_validation(knn_manhattan, X, y, k=10)
mean_accuracy3, accuracies3 = k_fold_cross_validation(knn_minkowski, X, y, k=10)

# Print Custom KNN Accuracies
print(f"KNN Mean Accuracies (Custom KNN):")
print(f"Using Euclidean distance: {mean_accuracy1 * 100}%")
print(f"Using Manhattan distance: {mean_accuracy2 * 100}%")
print(f"Using Minkowski distance: {mean_accuracy3 * 100}%")

# Sklearn KNN Models
knn1 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn2 = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn3 = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=3)

# Perform K-Fold Cross-Validation for Sklearn KNN
mean_accuracy_sklearn1, accuracies_sklearn1 = k_fold_cross_validation(knn1, X, y, k=10)
mean_accuracy_sklearn2, accuracies_sklearn2 = k_fold_cross_validation(knn2, X, y, k=10)
mean_accuracy_sklearn3, accuracies_sklearn3 = k_fold_cross_validation(knn3, X, y, k=10)

# Print Sklearn KNN Accuracies
print(f"\nKNN Mean Accuracies (Sklearn KNN):")
print(f"Using Euclidean distance: {mean_accuracy_sklearn1 * 100}%")
print(f"Using Manhattan distance: {mean_accuracy_sklearn2 * 100}%")
print(f"Using Minkowski distance: {mean_accuracy_sklearn3 * 100}%")

# T-test Results for Custom vs Sklearn KNN
t_stat1, p_val1 = custom_ttest_rel(accuracies1, accuracies_sklearn1)
t_stat2, p_val2 = custom_ttest_rel(accuracies2, accuracies_sklearn2)
t_stat3, p_val3 = custom_ttest_rel(accuracies3, accuracies_sklearn3)

alpha = 0.05

# Print T-test Results
print(f"\nT-test results for Custom KNN (Euclidean) vs Sklearn KNN (Euclidean):")
print(f"T-statistic: {t_stat1}, P-value: {p_val1}")
if p_val1 > alpha:
    print("The difference is not statistically significant. We reject the null hypothesis.")
else:
    print("The difference is statistically significant; we accept the alternative hypothesis.")

print(f"\nT-test results for Custom KNN (Manhattan) vs Sklearn KNN (Manhattan):")
print(f"T-statistic: {t_stat2}, P-value: {p_val2}")
if p_val2 > alpha:
    print("The difference is not statistically significant. We reject the null hypothesis.")
else:
    print("The difference is statistically significant; we accept the alternative hypothesis.")

print(f"\nT-test results for Custom KNN (Minkowski) vs Sklearn KNN (Minkowski):")
print(f"T-statistic: {t_stat3}, P-value: {p_val3}")
if p_val3 > alpha:
    print("The difference is not statistically significant. We reject the null hypothesis.")
else:
    print("The difference is statistically significant; we accept the alternative hypothesis.")
