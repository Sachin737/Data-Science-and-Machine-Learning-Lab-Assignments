# Name : Sachin Mahawar
# Roll no : b20129
# Lab Assignment 6

# Importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg as AR

df = pd.read_csv('daily_covid_cases.csv')   # Reading csv file

# (a)

# Splitting data into two parts with 65% for train data and 35% for test data
df_train = df.iloc[:int(df.shape[0] * 0.65), :]
df_test = df.iloc[int(df.shape[0] * 0.65):, :]

xticks = []  # Storing xticks for plot
for i in range(0, df.shape[0], 60):
    xticks.append(df['Date'][i])

# (a)
ar_model = AR(df_train['new_cases'], lags=5).fit()  # Fitting Auto regression model using training data
coef = ar_model.params  # Coefficients
print("Coefficients :", [a for a in coef])

# (b)

history = df_train.iloc[len(df_train) - 5:, 1].tolist()
predictions = []    # Storing predicted values

for i in range(df_test.shape[0]):   # Iterating over all test sample
    lag = []    # storing original previous 5 days data
    for p in range(len(history) - 5, len(history)):
        lag.append(history[p])
    y = coef[0]  # predictions
    for j in range(5):
        y += coef[j + 1] * lag[5 - j - 1]
    predictions.append(y)   # appending predictions
    history.append(df_test.iloc[i, 1])  # Appending previous given test sample

plt.figure(3)
plt.scatter(predictions, df_test.iloc[:, 1])    # Scatter plot bw predictions and original test values
plt.xlabel('Predicted values'), plt.ylabel('Actual values')
plt.show()

# (c)
plt.figure(4)
plt.plot(df_test.iloc[:, 0], predictions)   # Plotting line plot of predictions
plt.plot(df_test.iloc[:, 0], df_test.iloc[:, 1])  # Plotting line plot of original test data
plt.legend(['Predictions', 'Actual'])
plt.xticks(xticks[7:12])
plt.xlabel('Predicted values'), plt.ylabel('Actual values')
plt.show()

# (d)

# Computing RMSE bw original test data and predictions
RMSE = (np.sqrt(mean_squared_error(df_test.iloc[:,1], predictions))/np.mean(df_test.iloc[:,1]))*100
print("RMSE : ", RMSE, "%")
# Computing MAPE bw original test data and predictions
MAPE = (np.mean(np.abs((df_test.iloc[:,1]- predictions)/df_test.iloc[:,1])))*100
print("MAPE :",MAPE.round(3))