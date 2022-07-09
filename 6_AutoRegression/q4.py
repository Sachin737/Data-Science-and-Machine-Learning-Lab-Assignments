# Name : Sachin Mahawar
# Roll no : b20129
# Lab Assignment 6

# Importing Libraries
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
from statsmodels.tsa.ar_model import AutoReg as AR
from sklearn.metrics import mean_squared_error

df = pd.read_csv('daily_covid_cases.csv')  # Reading csv file

# Splitting data into two parts with 65% for train data and 35% for test data
df_train = df.iloc[:int(df.shape[0] * 0.65), :]
df_test = df.iloc[int(df.shape[0] * 0.65):, :]

p=1
while p < (df_train.shape[0]):
    corr = pearsonr(df_train.iloc[p:, 1].ravel(), df_train.iloc[:(df_train.shape[0]-p),1].ravel())[0].round(3)  # Computing Pearson correlation b/w Actual and lagged data
    if np.absolute(corr) <= 2/(len(df_train.iloc[p:, 1]))**0.5:
        print('The heuristic value is', p-1)
        break
    else:
        p += 1


z = p-1

ar_model = AR(df_train.iloc[:,1], lags=z, old_names=False).fit()   # Fitting Auto regression model using training data
coef = ar_model.params  # Coefficients

history = df_train.iloc[len(df_train) - z:, 1].tolist()
predictions = []    # Storing predicted values

for i in range(df_test.shape[0]):
    lag = []  # storing original previous 5 days data
    for p in range(len(history) - z, len(history)):
        lag.append(history[p])
    y = coef[0]  # predictions
    for j in range(z):
        y += coef[j + 1] * lag[z - j - 1]
    predictions.append(y)  # appending predictions
    history.append(df_test.iloc[i, 1])  # Appending previous given test sample

# Computing RMSE bw original test data and predictions
RMSE = (np.sqrt(mean_squared_error(df_test.iloc[:,1], predictions))/np.mean(df_test.iloc[:,1]))*100
print("RMSE : ", RMSE, "%")

# Computing MAPE bw original test data and predictions
MAPE = (np.mean(np.abs((df_test.iloc[:,1]- predictions)/df_test.iloc[:,1])))*100
print("MAPE :",MAPE.round(3))