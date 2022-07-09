# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

import pandas as pd

# Reading csv file
df = pd.read_csv("landslide_data3_miss.csv")

# Droping rows having blank cell in stationid attribute
df.drop(df[df['stationid'].isna()].index, inplace=True)

# Dropping all rows with more that 2 nan values6
df = df[df.isnull().sum(axis=1) <= 2]

# number of missing values in each row
a = df.isna().sum(axis=0)
print(a)

print("Total number of missing values", a.sum())

