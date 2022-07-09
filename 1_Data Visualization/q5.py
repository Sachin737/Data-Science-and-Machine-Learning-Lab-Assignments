# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

import matplotlib.pyplot as plt
import pandas as pd

# Reading csv file
data = pd.read_csv("pima-indians-diabetes.csv")

# Creating two Dataframe, one with class=0 and other with class=1
preg0 = data.loc[data["class"] == 0]
preg1 = data.loc[data["class"] == 1]


# Histogram for pregs having class=0
a = plt.figure(1)
plt.hist(preg0["pregs"], bins=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
plt.ylabel("frequency")
plt.xlabel("no of pregnancies{class=0}")
plt.show()

# Histogram for pregs having class=1
b = plt.figure(2)
plt.hist(preg1["pregs"], bins=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
plt.ylabel("frequency")
plt.xlabel("no of pregnancies{class=1}")
plt.show()
