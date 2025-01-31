import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv("data.csv")
data = data.drop_duplicates()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# Apply PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(X)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)
df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])

print("Explained variance ratio:", pca.explained_variance_ratio_)

# Plot PCA results
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=y, cmap='viridis')
plt.title('PCA - Reduced Dimensionality')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Class')
plt.show()

# Function for Parzen Window classification
def parzen_window_classification(X_train, y_train, X_test, bandwidth):
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_samples = X_test.shape[0]

    probabilities = np.zeros((n_samples, n_classes))

    for i, c in enumerate(classes):
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(X_train[y_train == c])

        log_prob = kde.score_samples(X_test)
        probabilities[:, i] = np.exp(log_prob)

    # Predict the class with the highest probability
    y_pred = np.argmax(probabilities, axis=1)
    return y_pred

# Define different bandwidths to test
bandwidths = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0]

# using pca values speed up the computation
# but decrease the accuracy
best_bandwidth = 0
best_accuracy = 0.00
# Evaluate Parzen Window classification for each bandwidth
for bandwidth in bandwidths:
    # Predict on the validation set

    #y_val_pred = parzen_window_classification(X_train, y_train, X_val, bandwidth)
    y_val_pred = parzen_window_classification(X_train_pca, y_train, X_val_pca, bandwidth)
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Bandwidth: {bandwidth}, Validation Accuracy: {val_accuracy:.4f}")
    if val_accuracy > best_accuracy and val_accuracy < 1.00:
        best_bandwidth = bandwidth
        best_accuracy = val_accuracy


# Evaluate on the test set using the best bandwidth
#y_test_pred = parzen_window_classification(X_train, y_train, X_test, best_bandwidth)
y_test_pred = parzen_window_classification(X_train_pca, y_train, X_test_pca, best_bandwidth)

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"\nBest Bandwidth: {best_bandwidth}, Test Accuracy: {test_accuracy:.4f}")
print(classification_report(y_test, y_test_pred))
