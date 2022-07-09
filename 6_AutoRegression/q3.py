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

# Splitting data into two parts with 65% for train data and 35
df_train = df.iloc[:int(df.shape[0] * 0.65), :]
df_test = df.iloc[int(df.shape[0] * 0.65):, :]

lagval = [1,5,10,15,25]     # Lag values
RMSE=[]   # Storing RMSE values for different Lag values
MAPE=[]   # Storing MAPE values for different Lag values

for i in lagval:    # Iterating over all lag values
    ar_model = AR(df_train['new_cases'], lags=i).fit()  # Fitting Auto regression model using training data
    coef = ar_model.params   # Coefficients

    history = df_train.iloc[len(df_train) - i:, 1].tolist()
    predictions = []    # Storing predicted values

    for k in range(df_test.shape[0]):   # Iterating over all test sample
        lag = []     # storing original previous i days data
        for p in range(len(history) - i, len(history)):
            lag.append(history[p])
        y = coef[0]  # predictions
        for j in range(i):
            y += coef[j + 1] * lag[i - j - 1]
        predictions.append(y)   # appending predictions
        history.append(df_test.iloc[k, 1])  # Appending previous given test sample

    rmse= (np.sqrt(mean_squared_error(df_test.iloc[:,1], predictions))/np.mean(df_test.iloc[:,1]))*100  # Computing RMSE bw original test data and predictions
    mape = (np.mean(np.abs((df_test.iloc[:,1]- predictions)/df_test.iloc[:,1])))*100    # Computing MAPE bw original test data and predictions

    RMSE.append(rmse)
    MAPE.append(mape.round(5))

plt.figure(1)
plt.bar(lagval, RMSE)   # Plotting bar plot of RMSE vs p.
plt.ylabel('RMSE'), plt.xlabel('Lag Values')
plt.xticks(lagval)


plt.figure(2)
plt.bar(lagval, MAPE)   # Plotting bar plot of MAPE vs p.
plt.ylabel('MAPE'), plt.xlabel('Lag Values')
plt.xticks(lagval)
plt.show()