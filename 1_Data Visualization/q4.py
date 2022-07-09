# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

import matplotlib.pyplot as plt
import pandas as pd

# Reading csv file
data = pd.read_csv("pima-indians-diabetes.csv")

# Histogram for pregs
a = plt.figure(1)
plt.hist(data["pregs"], color='r', bins=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
plt.title("Histogram(pregs)")
plt.xlabel("no of pregnancies")
plt.ylabel("frequency")
plt.show()

# Histogram for skin
b = plt.figure(2)
plt.hist(data["skin"], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.title("Histogram(skin)")
plt.xlabel("skin thickness")
plt.ylabel("frequency")
plt.show()
