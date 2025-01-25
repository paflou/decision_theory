import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import warnings
warnings.filterwarnings("ignore")

# Function to plot training, validation, and test errors for various degrees
def plot_errors(degrees, train_errors, val_errors, title):
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_errors, 'o-', color='blue', label='Training Error')
    plt.plot(degrees, val_errors, 'o-', color='red', label='Validation Error')
    plt.title(title)
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

# Function to train polynomial regression with regularization and track training/validation errors
def test_polynomial_fit_with_regularization(degrees, regularization='None', alpha=1.0):
    train_errors = []
    val_errors = []

    for degree in degrees:
        polynomial_features = PolynomialFeatures(degree=degree)
        X_train_poly = polynomial_features.fit_transform(X_train.drop(columns=['Date']))
        X_val_poly = polynomial_features.transform(X_val.drop(columns=['Date']))

        # Choose model based on regularization type
        if regularization == 'None':
            model = LinearRegression()
            title = 'Polynomial Regression without Regularization'
        elif regularization == 'L2':
            model = Ridge(alpha=alpha)
            title = f'Polynomial Regression with L2 Regularization (Ridge), alpha={alpha}'
        elif regularization == 'L1':
            model = Lasso(alpha=alpha, max_iter=100000)  # Lasso may require more iterations
            title = f'Polynomial Regression with L1 Regularization (Lasso), alpha={alpha}'

        # Fit the model
        model.fit(X_train_poly, y_train['Close'])

        # Predict on training and validation sets
        y_train_pred = model.predict(X_train_poly)
        y_val_pred = model.predict(X_val_poly)

        # Calculate MSE for training and validation sets
        train_errors.append(mean_squared_error(y_train['Close'], y_train_pred))
        val_errors.append(mean_squared_error(y_val['Close'], y_val_pred))

    # Plot the training and validation errors
    plot_errors(degrees, train_errors, val_errors, title)

    # Return the best degree based on validation error
    best_degree = degrees[np.argmin(val_errors)]
    print(f'Best degree based on validation error: {best_degree}')
    return best_degree

def evaluate_best_model(degree, regularization='None', alpha=1.0):
    polynomial_features = PolynomialFeatures(degree=degree)

    # Transform both training and test sets to include polynomial features
    X_train_poly = polynomial_features.fit_transform(X_train_full[['close_t-1', 'close_t-2', 'close_t-3','weeklyAvg']])  # Train on full training set
    X_test_poly = polynomial_features.transform(X_test[['close_t-1', 'close_t-2', 'close_t-3','weeklyAvg']])

    # Choose model based on regularization type
    if regularization == 'None':
        model = LinearRegression()
        title = 'Polynomial Regression without Regularization'
    elif regularization == 'L2':
        model = Ridge(alpha=alpha)
        title = f'Polynomial Regression with L2 Regularization (Ridge), alpha={alpha}'
    elif regularization == 'L1':
        model = Lasso(alpha=alpha, max_iter=100000)
        title = f'Polynomial Regression with L1 Regularization (Lasso), alpha={alpha}'

    # Fit the model on the full training set
    model.fit(X_train_poly, y_train_full['Close'])

    # Predict on the test set
    y_test_pred = model.predict(X_test_poly)

    # Calculate the test MSE
    test_error = mean_squared_error(y_test['Close'], y_test_pred)
    print(f'Test MSE with degree {degree} and {regularization} regularization: {test_error:.3f}')

    # Sort the test data by Date to ensure predictions are aligned with dates
    sorted_indices = np.argsort(X_test['Date'])
    X_test_sorted = X_test.iloc[sorted_indices]
    y_test_sorted = y_test.iloc[sorted_indices]
    y_test_pred_sorted = y_test_pred[sorted_indices]

    plt.figure(figsize=(10, 6))

    # Plot the training data
    plt.scatter(X_train_full['Date'], y_train_full['Close'], color='blue', label='Training Data', s=5)
    # Plot the test data
    plt.scatter(X_test_sorted['Date'], y_test_sorted['Close'], color='red', label='Test Data', s=5)
    # Plot the model prediction curve
    plt.plot(X_test_sorted['Date'], y_test_pred_sorted, color='green', label=f'Polynomial Degree {degree}', linewidth='1')

    plt.title(title)
    plt.grid(True)
    plt.xlabel('Time (DateNum)')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()



#######################################################


raw_data = pd.read_csv('close_prices.csv')
raw_data['Date'] = pd.to_datetime(raw_data['Date'])

close_prices = raw_data['Close']
smoothed_prices = gaussian_filter1d(close_prices, sigma=1)
raw_data['Close'] = smoothed_prices

raw_data['close_t-1'] = raw_data['Close'].shift(-1)  # Price from the previous day
raw_data['close_t-2'] = raw_data['Close'].shift(-2)  # Price from the previous day
raw_data['close_t-3'] = raw_data['Close'].shift(-3)  # Price from the previous day
raw_data['weeklyAvg'] = raw_data['Close'].rolling(window=7, min_periods=1).mean()  # 7-day moving average

raw_data = raw_data.dropna()  # Drop rows with missing values

X = raw_data[['Date', 'close_t-1','close_t-2', 'close_t-3','weeklyAvg']]  # Feature (numeric days) as X
y = raw_data[['Close','Date']]   # Target (Close prices) as y

#correlations = raw_data.corr()
#print(correlations['Close'])

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=0)

# Test different degrees of polynomial without regularization
degrees = list(range(1, 4))
print('Testing different degrees of polynomial without regularization')
best_degree_no_reg = test_polynomial_fit_with_regularization(degrees, regularization='None')

# Evaluate the best model on the test set (without regularization)
evaluate_best_model(degree=best_degree_no_reg, regularization='None')


print('Testing different degrees of polynomial with L2 regularization')
# Test different degrees of polynomial with L2 regularization (Ridge)
best_degree_l2 = test_polynomial_fit_with_regularization(degrees, regularization='L2', alpha=1.0)

# Evaluate the best model on the test set (with L2 regularization)
evaluate_best_model(degree=best_degree_l2, regularization='L2', alpha=1.0)


print('Testing different degrees of polynomial with L1 regularization')
# Test different degrees of polynomial with L1 regularization (Lasso)
best_degree_l1 = test_polynomial_fit_with_regularization(degrees, regularization='L1', alpha=0.25)

# Evaluate the best model on the test set (with L1 regularization)
evaluate_best_model(degree=best_degree_l1, regularization='L1', alpha=0.25)
