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

# Function to replace outliers with median
def Replace(lst):
    lst = lst.tolist()
    q1, q3 = np.percentile(lst, [25, 75])
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    med = np.median(lst)

    for i in range(len(lst)):
        if (lst[i] < lower_bound) or (lst[i] > upper_bound):
            lst[i] = med
    return lst


temp_new = Replace(df['temperature'])
rain_new = Replace(df['rain'])


# Boxplot for temperature
p1 = plt.figure(1)
plt.boxplot(temp_new)
plt.title("Box Plot of temperature after replacing the outliers")
plt.xlabel("Temperature")
plt.ylabel("Values")

# Boxplot for rain
p2 = plt.figure(2)
plt.boxplot(rain_new)
plt.title("Box Plot of rain after replacing the outliers")
plt.xlabel("Rain")
plt.ylabel("Values")

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


temp_outliers = outliers(temp_new)
rain_outliers = outliers(rain_new)

print("Outliers of attribute temperature : ", temp_outliers,'\n\n')
print("Outliers of attribute rain : ", rain_outliers)

print(min(rain_outliers))
print(max(rain_outliers))


