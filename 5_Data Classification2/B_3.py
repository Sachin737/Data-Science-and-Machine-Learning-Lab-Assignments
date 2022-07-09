#  Name : Sachin Mahawar
# # Roll number : B20129
# # Mobile Number : 9166843951

# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# Reading csv files
df = pd.read_csv('abalone.csv')

# Splitting whole data set with training data containing 70% data from each class and test data with containing rest 30% data from each class
X_train, X_test, X_label_train, X_label_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], random_state=42, test_size=0.30)

maxi = 0  # Maximum pearson correlation coefficient
attr = ""  # attribute with pearson correlation coefficient with Rings attribute
for i in df.columns:
    if i != "Rings":
        g = pearsonr(df[i], df['Rings'])[0]
        if g > maxi:
            maxi = g
            attr = i

# (a)
X_train = X_train[attr].to_numpy().reshape(-1, 1)  # X train column having maximum pearson correlation coefficient

train_rmse = []
P_val = [2, 3, 4, 5]
for p in P_val:
    # Transforming X_train into their corresponding polynomial features
    poly = PolynomialFeatures(degree=p)
    X_train_poly = poly.fit_transform(X_train)

    # Fitting linear regression on polynomial dataset
    Lin_reg = LinearRegression()
    Lin_reg.fit(X_train_poly, X_label_train)

    # Predictions
    train_pred = Lin_reg.predict(X_train_poly)

    rmse = mean_squared_error(X_label_train, train_pred) ** 0.5  # Calculating RMSE
    train_rmse.append(rmse)
    print("Rmse for p=", p, ':', round(rmse, 3))

# bar graph of RMSE (y-axis) vs different values of degree of the polynomial (x-axis).
a = plt.figure(1)
plt.bar(P_val, train_rmse, width=0.3)
plt.title('Univariate non-linear regression model')
plt.xlabel('Degree of Polynomial (p)')
plt.ylabel('Train RMSE')
plt.xticks(P_val)
plt.show()

# (b)
print("(b)")

X_test = X_test[attr].to_numpy().reshape(-1, 1)  # X test column having maximum pearson correlation coefficient

test_rmse = []
for p in P_val:
    # Transforming X_train ans X_test into their corresponding polynomial features
    poly = PolynomialFeatures(degree=p)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)

    # Fitting linear regression on polynomial dataset
    Lin_reg = LinearRegression()
    Lin_reg.fit(X_train_poly, X_label_train)

    # Predictions
    test_pred = Lin_reg.predict(X_test_poly)

    rmse = mean_squared_error(X_label_test, test_pred) ** 0.5  # Calculating RMSE
    test_rmse.append(rmse)
    print("Rmse for p=", p, ':', round(rmse, 3))

# bar graph of RMSE (y-axis) vs different values of degree of the polynomial (x-axis).
b = plt.figure(2)
plt.title("Univariate non-linear regression model")
plt.bar(P_val, test_rmse, width=0.3)
plt.xlabel('Degree of Polynomial (p)')
plt.ylabel('Test RMSE')
plt.xticks(P_val)
plt.show()

# (c)

# p=5 has the lowest rmse for test data
# So finding predictions for p=5
poly5 = PolynomialFeatures(degree=5)
X_train_poly5 = poly5.fit_transform(X_train)
Lin_reg = LinearRegression()
Lin_reg.fit(X_train_poly5, X_label_train)
train_pred = Lin_reg.predict(X_train_poly5)

# Plotting best fit line on the training data
c = plt.figure(3)
plt.title("Best Fit Curve")
plt.scatter(X_train, X_label_train)
plt.scatter(X_train, train_pred, color='r', marker='.')
plt.xlabel('Shell weight')
plt.ylabel('Rings')
plt.title('Best Fit Curve')
plt.show()

# (d)
# scatter plot of the actual number of Rings (x-axis) vs the predicted number of
# Rings (y-axis) on the test data

d = plt.figure(4)
plt.scatter(X_label_test, test_pred, marker=".")
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Univariate non-linear regression model')
plt.show()
