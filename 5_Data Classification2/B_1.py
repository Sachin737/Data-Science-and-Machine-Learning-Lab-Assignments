# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# Reading csv files
df = pd.read_csv('abalone.csv')

# Splitting whole data set with training data containing 70% data from each class and test data with containing rest 30% data from each class
X_train, X_test, X_label_train, X_label_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], random_state=42, test_size=0.30)

# Merging Ring attribute into Train and test data to create new csv file
train_merged = pd.concat([X_train, X_label_train], axis=1, join="inner")
test_merged = pd.concat([X_test, X_label_test], axis=1, join="inner")
train_merged.to_csv("abalone-train.csv")
test_merged.to_csv("abalone-test.csv")

maxi = 0  # Maximum pearson correlation coefficient
attr = ""  # attribute with pearson correlation coefficient with Rings attribute
for i in df.columns:
    if i != "Rings":
        g = pearsonr(df[i], df['Rings'])[0]
        if g > maxi:
            maxi = g
            attr = i

train = X_train.copy()
test = X_test.copy()

X_train = X_train[attr].to_numpy().reshape(-1, 1)  # X train column having maximum pearson correlation coefficient
X_test = X_test[attr].to_numpy().reshape(-1, 1)  # X test column having maximum pearson correlation coefficient

Lin_reg = LinearRegression()  # Creating regression object
Lin_reg.fit(X_train, X_label_train)  # Training regressor

# Plotting best fit line on the training data
plt.title("Best Fit Line")
plt.scatter(X_train, X_label_train, marker='.', color='b', label="Training Points")  # Scatter plot of train data
plt.plot(train[attr], Lin_reg.predict(X_train), color='r', label="Regression line")  # plotting predicted line
plt.legend()
plt.ylabel('Rings')
plt.xlabel(attr)
plt.show()

# Calculating RMSE
RMSE_train = mean_squared_error(X_label_train, Lin_reg.predict(X_train)) ** 0.5
RMSE_test = mean_squared_error(X_label_test, Lin_reg.predict(X_test)) ** 0.5
print(" RMSE (train):", RMSE_train)
print(" RMSE (test):", RMSE_test)

# scatter plot of actual Rings (x-axis) vs predicted Rings (y-axis) on the test data
plt.figure(2)
plt.title("Univariate linear regression model")
plt.scatter(X_label_test, Lin_reg.predict(X_test), marker='.')
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.show()

