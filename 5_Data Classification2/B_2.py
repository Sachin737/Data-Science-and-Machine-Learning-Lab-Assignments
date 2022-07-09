# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Reading csv files
df = pd.read_csv('abalone.csv')

# Splitting whole data set with training data containing 70% data from each class and test data with containing rest 30% data from each class
X_train, X_test, X_label_train, X_label_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], random_state=42, test_size=0.30)

Lin_reg = LinearRegression()  # Creating regression object
Lin_reg.fit(X_train, X_label_train)  # Training regressor

# Plotting best fit line on the training data
plt.title("Multivariate linear regression model")
plt.scatter(X_label_test, Lin_reg.predict(X_test), marker='.', color='b', label="Training Points")  # scatter plot of actual Rings vs predicted Rings on the test data
plt.legend()
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.show()

RMSE_train = mean_squared_error(X_label_train, Lin_reg.predict(X_train)) ** 0.5  # RMSE value for train data
RMSE_test = mean_squared_error(X_label_test, Lin_reg.predict(X_test)) ** 0.5  # RMSE value for test data
print("RMSE (train):", RMSE_train)
print("RMSE (test):", RMSE_test)
