import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

best_k = 0
best_accuracy = 0.00

for k in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    print(f'k={k}: Accuracy: {accuracy * 100:.2f}%')

    # Update best_k if accuracy improves (but is not 100%)
    if accuracy > best_accuracy and accuracy < 1.0:
        best_accuracy = accuracy
        best_k = k



print('\nTesting:')
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'k={best_k}: Test accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))
