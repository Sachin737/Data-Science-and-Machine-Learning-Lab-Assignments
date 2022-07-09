import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics

data = pd.read_csv("landslide_data3_miss.csv")
data2 = pd.read_csv("landslide_data3_original.csv")

vals = ['temperature', 'humidity', 'pressure',
        'rain', 'lightavgw/o0', 'lightmax', 'moisture']
data = data.interpolate(method='linear')

mode1 = []
for i in vals:
    mode1.append(statistics.mode(data[i]))

mean1 = data.mean()
median1 = data.median()
std1 = data.std()
mean2 = data2.mean()
median2 = data2.median()
std2 = data2.std()
a=pd.DataFrame([mean1,median1,mode1,std1],index=["mean1","median1","mode","standard deviation1"])
b=pd.DataFrame([mean2,median2,std2],index=["mean2","median2","standard deviation2"])
print("the values of mean, median and standard deviation of missing value file is :")

print(a)
print("the values of mean, median and standard deviation of original file is :")
print(b)