# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

import pandas as pd


# Function to compute mean
def mean(lst):
    sum = 0
    for i in lst:
        sum = sum + i
    return sum / len(lst)


# Function to compute standard deviation
def stDeviation(lst):
    var = 0
    for i in range(len(lst)):
        var += ((lst[i] - mean(lst)) ** 2) / (len(lst) - 1)

    SD = var ** 0.5
    return SD


# Function to compute correlation
def correlation(x, y):
    Xmean = mean(x)
    Ymean = mean(y)
    Covariance = 0
    for i in range(len(x)):
        Covariance += ((x[i] - Xmean) * (y[i] - Ymean)) / (len(x) - 1)

    SDx = stDeviation(x)
    SDy = stDeviation(y)
    Correlation = Covariance / (SDx * SDy)

    return Correlation


# Reading csv file
data = pd.read_csv("pima-indians-diabetes.csv")

# Printing correlation of BMI with all other attributes except "class"
for i in data.columns:
    if i == "class":
        break
    print(f"Correlation coefficient of BMI and {i} = ", correlation(data["BMI"], data[i]))
