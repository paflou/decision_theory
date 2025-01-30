import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Load data
data = pd.read_csv("data.csv")

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

best_k = 1
best_accuracy = 0

for k in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    print(f'k={k}: Validation Accuracy: {accuracy * 100:.2f}%')

    # Track the best k based on validation accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f'\nBest k: {best_k} with Validation Accuracy: {best_accuracy * 100:.2f}%')

# Train the final model with the best k on the combined training and validation set
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)  # You can also combine X_train and X_val if needed

# Evaluate on the test set
y_test_pred = final_knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'\nTest Accuracy with k={best_k}: {test_accuracy * 100:.2f}%')
