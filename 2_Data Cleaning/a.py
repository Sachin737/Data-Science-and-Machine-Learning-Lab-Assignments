import pandas as pd

df = pd.read_csv('landslide_data3_miss.csv')

count = []

attributes = ['temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']

for i in attributes:
    count.append(df[i].isna().sum())

print(count)