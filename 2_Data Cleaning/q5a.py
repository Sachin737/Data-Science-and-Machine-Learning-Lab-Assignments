# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading csv file
df = pd.read_csv("landslide_data3_miss.csv")

# Replacing the missing values in each attribute using linear interpolation technique.
df = df.interpolate(method='linear')

# Boxplot for temperature
p1 = plt.figure(1)
plt.boxplot(df['temperature'])
plt.title("Boxplot of 'temperature'")

# Boxplot for rain
p2 = plt.figure(2)
plt.boxplot(df['rain'])
plt.title("Boxplot of 'rain'")
plt.show()

# Funtion to find outliers
def outliers(lst):
    q1, q3 = np.percentile(lst, [25, 75])
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    out = []
    for i in lst:
        if (i < lower_bound) or (i > upper_bound):
            out.append(i)
    return out


temp_outliers = outliers(df['temperature'])
rain_outliers = outliers(df['rain'])

print("Outliers of attribute temperature : ", temp_outliers,'\n\n')
print("Outliers of attribute rain : ", rain_outliers)

print(min(rain_outliers))
print(max(rain_outliers))