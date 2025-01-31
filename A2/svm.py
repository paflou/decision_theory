import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC


# Load data
data = pd.read_csv("data.csv")
data = data.drop_duplicates()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]



# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


linear_svm = SVC(kernel='linear')
param_grid_linear = {'C': [0.1, 1, 10, 100]}

grid_search_linear = GridSearchCV(linear_svm, param_grid_linear, cv=5, scoring='accuracy')
grid_search_linear.fit(X_train, y_train)

print("Best parameters:", grid_search_linear.best_params_)
# Test linear SVM
y_pred_linear = grid_search_linear.predict(X_test)
test_accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f"Test accuracy (Linear SVM): {test_accuracy_linear * 100:.2f}%")
print(classification_report(y_test, y_pred_linear))

# Non-linear SVM
print("\nNon linear SVM: ")
param_grid = [
    {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]},
    {'kernel': ['poly'], 'C': [0.1, 1, 10, 100], 'degree': [2, 3, 4]},
    {'kernel': ['sigmoid'], 'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}
]

# Grid search for best hyperparameters
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
# Test non-linear SVM
y_pred_non_linear = grid_search.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_non_linear)
print(f"Test Accuracy (Non-Linear SVM): {test_accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred_non_linear))
