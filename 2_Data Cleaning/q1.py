# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

import matplotlib.pyplot as plt
import pandas as pd

# Reading csv file
data = pd.read_csv("landslide_data3_miss.csv")

# List of all the attributes given in data
attributes = ['dates', 'stationid', 'temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']

# List of count of nan for all attributes
missing_count = []

for i in range(9):
    a = data[attributes[i]].isna().sum()
    missing_count.append(a)

# Changing font size
plt.rcParams.update({'font.size': 6.5})

# Bar chart
plt.grid(zorder=0)
plt.bar(attributes, missing_count,zorder=2)
plt.ylabel("Nan-count",fontsize=10)
plt.show()
