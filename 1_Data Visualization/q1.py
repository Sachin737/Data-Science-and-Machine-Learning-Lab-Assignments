# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

import math
import pandas as pd


# functions to compute mean, median, mode, maximum, minimum and standard deviation
def mean(lst):
    sum = 0
    for i in lst:
        sum = sum + i
    return sum / len(lst)


def median(lst):
    lst = sorted(lst)
    a = len(lst)

    if (a%2 == 0):
        return float((lst[int(a / 2)] + lst[int(a / 2) - 1]) / 2)
    else:
        return float(lst[int(a / 2) + 1])


def mode(lst):
    freq = {}
    for i in lst:
        if i in freq:
            freq[i] += 1
        else:
            freq[i] = 1

    mx = max(freq.values())
    return [k for k, v in freq.items() if v == mx]




def maximum(lst):
    max = 0
    for i in lst:
        if i > max:
            max = i
    return max


def minimum(lst):
    mini = math.inf

    for i in lst:
        if i < mini:
            mini = i
    return mini


def stDeviation(lst):
    var = 0
    for i in range(len(lst)):
        var += ((lst[i] - mean(lst)) ** 2) / (len(lst)-1)

    SD = var ** (0.5)
    return SD


# reading csv file
df = pd.read_csv("pima-indians-diabetes.csv")

# printing values of mean, median, mode, maximum, minimum, standard deviation for all attributes
for i in df.columns:
    if i == "class":
        break
    print(f"For {i}")
    print(f"Mean :", mean(df[i]))
    print(f"Median :", median(df[i]))
    print(f"Mode :", mode(df[i]))
    print(f"Maximum :", maximum(df[i]))
    print(f"Minimum :", minimum(df[i]))
    print(f"Standard Deviation :", stDeviation(df[i]))
    print("\n")


