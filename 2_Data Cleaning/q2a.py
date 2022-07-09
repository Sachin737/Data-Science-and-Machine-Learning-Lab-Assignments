# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

import pandas as pd

# Reading csv file
df = pd.read_csv("landslide_data3_miss.csv")

# Counting number of blank cell in stationid atribute
count = df['stationid'].isna().sum()

# Droping rows having blank cell in stationid attribute
df.drop(df[df['stationid'].isna()].index, inplace=True)

print("Total number of rows deleted: ", count)

