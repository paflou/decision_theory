import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("data.csv")

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apply PCA for visualization (optional)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(X)
df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])

print("Explained variance ratio:", pca.explained_variance_ratio_)

# Plot PCA results
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=y, cmap='viridis')
plt.title('PCA - Reduced Dimensionality')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Class')
plt.show()

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Function for Parzen Window classification
def parzen_window_classification(X_train, y_train, X_test, bandwidth):
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_samples = X_test.shape[0]

    # Initialize probabilities
    probabilities = np.zeros((n_samples, n_classes))

    for i, c in enumerate(classes):
        # Train KernelDensity for each class
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(X_train[y_train == c])

        # Calculate log probabilities for each sample
        log_prob = kde.score_samples(X_test)
        probabilities[:, i] = np.exp(log_prob)

    # Predict the class with the highest probability
    y_pred = np.argmax(probabilities, axis=1)
    return y_pred

# Define different bandwidths to test
bandwidths = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]

# Evaluate Parzen Window classification for each bandwidth
for bandwidth in bandwidths:
    # Predict on the validation set
    y_val_pred = parzen_window_classification(X_train, y_train, X_val, bandwidth)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Bandwidth: {bandwidth}, Validation Accuracy: {val_accuracy:.4f}")

# Choose the best bandwidth based on validation performance
best_bandwidth = 0.5

# Evaluate on the test set using the best bandwidth
y_test_pred = parzen_window_classification(X_train, y_train, X_test, best_bandwidth)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Best Bandwidth: {best_bandwidth}, Test Accuracy: {test_accuracy:.4f}")
