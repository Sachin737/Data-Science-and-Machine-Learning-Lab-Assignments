# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

import pandas as pd
import numpy as np
import statistics

# Reading csv file
df = pd.read_csv("landslide_data3_miss.csv")
df_o = pd.read_csv('landslide_data3_original.csv')

# List of attributes expect date and stationid
attributes = ['temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']

######################################################################################################
# Computing mean ,median.mode and st deviation for all attributres before replacing
print("Before replacing")

o_mean = []
for i in attributes:
    o_mean.append(np.mean(df_o[i]))

o_median = []
for i in attributes:
    o_median.append(np.median(df_o[i]))

o_mode = []
for i in attributes:
    o_mode.append(statistics.mode(df_o[i]))

o_stdev = []
for i in attributes:
    o_stdev.append(np.std(df_o[i]))

o_ans = pd.DataFrame(list(zip(o_mean, o_median, o_mode, o_stdev)), index=attributes,
                     columns=['Mean', 'Median', 'Mode', 'St Deviation'])
print(o_ans)

######################################################################################################
# Computing mean ,median.mode and st deviation for all attributres after replacing
print("After replacing")

# Replacing nan with mean of that attribute
for i in attributes:
    df.fillna({i: np.mean(df[i])}, inplace=True)

d_mean = []
for i in attributes:
    d_mean.append(np.mean(df[i]))

d_median = []
for i in attributes:
    d_median.append(np.median(df[i]))

d_mode = []
for i in attributes:
    d_mode.append(statistics.mode(df[i]))

d_stdev = []
for i in attributes:
    d_stdev.append(np.std(df[i]))

d_ans = pd.DataFrame(list(zip(d_mean, d_median, d_mode, d_stdev)), index=attributes,
                     columns=['Mean', 'Median', 'Mode', 'St Deviation'])
print(d_ans, '\n\n')
