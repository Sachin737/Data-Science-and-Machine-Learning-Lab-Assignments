# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Reading csv files
df = pd.read_csv('abalone.csv')

# Splitting whole data set with training data containing 70% data from each class and test data with containing rest 30% data from each class
X_train, X_test, X_label_train, X_label_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], random_state=42, test_size=0.30)

# (a)
print("(a)")

P_val = [2, 3, 4, 5]
train_rmse = []
for p in P_val:
    # Transforming X_train into their corresponding polynomial features
    poly = PolynomialFeatures(degree=p)
    X_train_poly = poly.fit_transform(X_train)

    # Fitting linear regression on polynomial dataset
    Lin_reg = LinearRegression()
    Lin_reg.fit(X_train_poly, X_label_train)

    # Predictions
    train_pred = Lin_reg.predict(X_train_poly)

    # Calculating RMSE
    rmse = mean_squared_error(X_label_train, train_pred) ** 0.5
    train_rmse.append(rmse)
    print("Rmse for p=", p, ':', round(rmse, 3))

# bar graph of RMSE of training data vs different values of degree of the polynomial
a = plt.figure(1)
plt.title("Multivariate non-linear regression model")
plt.bar(P_val, train_rmse, width=0.3)
plt.xlabel('Degree of Polynomial (p)')
plt.xticks(P_val)
plt.ylabel('Train RMSE')
plt.show()

# (b)
print("(b)")
test_rmse = []
for p in P_val:
    # Transforming X_train and X_test into their corresponding polynomial features
    poly = PolynomialFeatures(degree=p)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)

    # Fitting linear regression on polynomial dataset
    Lin_reg = LinearRegression()
    Lin_reg.fit(X_train_poly, X_label_train)

    # Predictions
    test_pred = Lin_reg.predict(X_test_poly)

    # Calculating RMSE
    rmse = mean_squared_error(X_label_test, test_pred) ** 0.5
    test_rmse.append(rmse)
    print("Rmse for p=", p, ':', round(rmse, 3))

# bar graph of RMSE of test data vs different values of degree of the polynomial
b = plt.figure(2)
plt.title("Multivariate non-linear regression model")
plt.bar(P_val, test_rmse, width=0.3)
plt.xlabel('Degree of Polynomial (p)')
plt.xticks(P_val)
plt.ylabel('Test RMSE')
plt.show()

# (c)

# Best value of p is 2
poly = PolynomialFeatures(degree=2)
Xt = poly.fit_transform(X_train)
Xte = poly.fit_transform(X_test)
Lin_reg = LinearRegression()
Lin_reg.fit(Xt, X_label_train)
testpred = Lin_reg.predict(Xte)

# scatter plot of the actual number of Rings (x-axis) vs the predicted number of Rings (y-axis) on the test data
d = plt.figure(3)
plt.title("Multivariate non-linear regression model")
plt.scatter(X_label_test, testpred, marker="x")
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Multivariate non-linear regression model')
plt.show()
