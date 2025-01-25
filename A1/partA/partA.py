import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from test import *

raw_data = pd.read_csv('close_prices.csv')
raw_data['Date'] = pd.to_datetime(raw_data['Date'])
raw_data = raw_data[raw_data['Date'].dt.year > 2020]  # Δεδομένα εκπαίδευσης (πριν το 2024)

#print(raw_data.head())
close_prices = raw_data['Close']

# higher sigma value will result in a smoother curve
# and thus a less accurate prediction
# and error metrics will be lower
smoothed_prices = gaussian_filter1d(close_prices, sigma=1)
raw_data['Close'] = smoothed_prices

raw_data['close_t-1'] = raw_data['Close'].shift(-1)  # Price from the previous day
raw_data['close_t-2'] = raw_data['Close'].shift(-2)  # Price from the previous day
raw_data['close_t-3'] = raw_data['Close'].shift(-3)  # Price from the previous day
raw_data['weeklyAvg'] = raw_data['Close'].rolling(window=7, min_periods=1).mean()  # 7-day moving average
#print(raw_data.head())

raw_data = raw_data.dropna()  # Drop rows with missing values

train_data = raw_data[raw_data['Date'].dt.year < 2024]  # Δεδομένα εκπαίδευσης (πριν το 2024)
val_data = raw_data[raw_data['Date'].dt.year == 2024]   # Δεδομένα επικύρωσης (2024)



exog = sm.add_constant(train_data[['close_t-1','weeklyAvg','close_t-2', 'close_t-3']])  # Combine both variables
model_linear_regression = sm.OLS(
    endog = train_data['Close'],
    exog = exog)
results_regression = model_linear_regression.fit()

print(results_regression.summary())
print(results_regression.params)

validationData_y = sm.add_constant(val_data[['close_t-1','weeklyAvg','close_t-2', 'close_t-3']])  # Combine both variables
prediction = results_regression.predict(validationData_y)

trainData = sm.add_constant(train_data[['close_t-1','weeklyAvg','close_t-2', 'close_t-3']])  # Combine both variables
predictionTrain = results_regression.predict(trainData)

true_values = val_data['Close']
test_values = train_data['Close']

# Calculate RMSE, MSE, MAE and print them
rmse_train = rmse(test_values, predictionTrain)
mae_train = mean_absolute_error(test_values, predictionTrain)
mse_train = mean_squared_error(test_values, predictionTrain)

print("Root Mean Square Error on train data:", rmse_train)
print("Mean Absolute Error on train data:", mae_train)
print("Mean Squared Error on train data:", mse_train)

# Calculate RMSE, MSE, MAE and print them
rmse = rmse(true_values, prediction)
mae = mean_absolute_error(true_values, prediction)
mse = mean_squared_error(true_values, prediction)

print("Root Mean Square Error on validation data:", rmse)
print("Mean Absolute Error on validation data:", mae)
print("Mean Squared Error on validation data:", mse)


plt.scatter(val_data['Close'], prediction, label="Predicted Values", marker='o', color='red')
plt.plot(val_data['Close'], val_data['Close'], color='black', linewidth=1)
plt.xlabel("Actual Close Values")
plt.ylabel("Predicted Close Values")
plt.legend()
plt.title("Predicted vs Actual Values")
plt.show()

# Plot actual and predicted values against date
plt.plot(val_data['Date'], val_data['Close'], label='Actual', color='blue')
plt.plot(val_data['Date'], prediction, label='Prediction', linestyle='dashed', color='red')

# Customize the plot
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices in 2024')
plt.legend()
plt.xticks(rotation=30)
plt.grid(True)
plt.show()


fullData_y = sm.add_constant(raw_data[['close_t-1','weeklyAvg','close_t-2', 'close_t-3']])  # Combine both variables
full_prediction = results_regression.predict(fullData_y)

# Plot actual and predicted values against date
plt.plot(raw_data['Date'], raw_data['Close'], label='Actual', color='blue')
plt.plot(raw_data['Date'], full_prediction, label='Prediction', linestyle='dashed', color='red')

# Customize the plot
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices since 2005')
plt.legend()
plt.xticks(rotation=30)
plt.grid(True)
plt.show()


def predict_linear_regression(fitted_model, dict_features):
    """
    Calculates y ~ const + sum( parameter*value )

    { 'feature name' : value }

    Does not assume you have all features present, so prediction may be off.
    Assumes const parameter is not present in dictionary
    """
    list_given_terms = [
        fitted_model.params[key]*value for key, value in dict_features.items()
    ]
    constant_value = fitted_model.params['const']
    list_given_terms.append(constant_value)

    return sum(list_given_terms)

def predict_tomorrow(fitted_model):
    # Extract the latest values
    close_t_1_value = raw_data['close_t-1'].iloc[0]
    weekly_avg_value = raw_data['weeklyAvg'].iloc[0]
    close_t_2_value = raw_data['close_t-2'].iloc[0]
    close_t_3_value = raw_data['close_t-3'].iloc[0]

    latest_date = pd.to_datetime(raw_data['Date'].iloc[0])
    tomorrows_date = latest_date + pd.Timedelta(days=1)


    # Make the prediction
    prediction = predict_linear_regression(
        fitted_model,
        {'close_t-1': close_t_1_value, 'weeklyAvg': weekly_avg_value, 'close_t-2': close_t_2_value, 'close_t-3': close_t_3_value}
    )

    print(f"Predicted value for tomorrow ({tomorrows_date.strftime('%Y-%m-%d')}): {prediction}")


predict_tomorrow(results_regression)
