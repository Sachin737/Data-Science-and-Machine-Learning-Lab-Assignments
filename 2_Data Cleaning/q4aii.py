# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to find RMSE value
def RMSE(col, Na, idx):
    ans = 0
    for i in range(0, Na):
        ans += (np.mean(df[col]) - df_o[col][idx[i]]) ** 2
    return (ans / Na) ** 0.5


# Reading csv file
df = pd.read_csv("landslide_data3_miss.csv")
df_o = pd.read_csv('landslide_data3_original.csv')

# List of attributes expect date and stationid
attributes = ['temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']

nan_count = []
for i in attributes:
    nan_count.append(df[i].isna().sum())

# Storing index of nan in landslide_data3_miss.csv file
idx=[]
for i in range(7):
    a = df[df[attributes[i]].isna()].index
    idx.append(a)


# Replacing nan with mean of that attribute
for i in attributes:
    df.fillna({i: np.mean(df[i])}, inplace=True)


rmse_val = []
for i in range(7):
    rmse_val.append(RMSE(attributes[i], nan_count[i], idx[i]))
answer = pd.DataFrame(list(rmse_val), index=attributes, columns=['RMSE VALUES'])
print(answer)

# Changing font size
plt.rcParams.update({'font.size': 6.5})

# PLotting rmse value of all attributes
plt.bar(attributes, rmse_val)
plt.ylabel("RMSE value")
plt.xlabel('attributes')
plt.yscale('log')
plt.show()