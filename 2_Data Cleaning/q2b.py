# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

import pandas as pd

# Reading csv file
df = pd.read_csv("landslide_data3_miss.csv")

# Number of rows initially
rowi = df.shape[0]

# Dropping all rows with more that 2 nan values
df = df[df.isnull().sum(axis=1) < 3]

# Printing number of tuples deleted
print('Total tuples deleted: ',rowi - df.shape[0])

