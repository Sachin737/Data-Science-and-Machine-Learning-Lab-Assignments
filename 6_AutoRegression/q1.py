# Name : Sachin Mahawar
# Roll no : b20129
# Lab Assignment 6

# Importing libraries
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_csv('daily_covid_cases.csv')  # Reading csv file

# (a)
xticks = []     # Storing xticks for plot
for i in range(0, df.shape[0], 60):
    xticks.append(df['Date'][i])

plt.figure(1)
plt.title('Original data')
plt.plot(df['Date'], df['new_cases'], color='r')    # Plotting line plot of given data(new cases vs date)
plt.xticks(xticks)
plt.xlabel('Dates')
plt.ylabel('Confirmed new cases')

df['new_cases_shift'] = df['new_cases'].shift(1)    # Shifting data by one day(p=1)
df = df.dropna()    # Dropping nan value

# (b)
# plt.figure(2)
plt.title('1 Day lagged')
plt.plot(df['Date'], df['new_cases_shift'], color='b')    # Plotting line plot of shifted data(shifted data vs date)
plt.xticks(xticks)
plt.xlabel('Dates')
plt.ylabel('Confirmed new cases')
plt.show()

pcorr = pearsonr(df['new_cases'], df['new_cases_shift'])[0].round(3)    # Finding Pearson correlation coefficient bw original and one day shifted data
print("Pearson corr coefficient for one day shift: ", pcorr)

# (c)
plt.figure(3)
plt.scatter(df['new_cases'], df['new_cases_shift'], marker='.')     # Scatter plot of original data vs one day shifted data
plt.title('Original vs 1-day lagged')
plt.xlabel('Original'), plt.ylabel('1 day lagged')
plt.show()

# (d)
corr_coeff = []     # for storing correlation coefficient
lags = [1, 2, 3, 4, 5, 6]   # Lag values(p)

for i in lags:
    a = df['new_cases'].shift(i)    # Shifting data by i days (p=i)
    a = a.dropna()
    corr = pearsonr(df.iloc[i:, 1], a)[0]   # Finding correlation coefficient
    corr_coeff.append(corr)     # Appending corrr coef

plt.figure(4)
plt.plot(range(1, 7), corr_coeff, ls=':')   # Plotting line plot for correlation coefficient
plt.xlabel('lagged value(p)'),plt.ylabel('Pearson Correlation')
plt.show()

# (e)
plot_acf(df['new_cases'])   # Plotting auto correlation plot for different p values.
plt.xlabel('lagged value(p)'),plt.ylabel('Pearson Correlation')
plt.show()
