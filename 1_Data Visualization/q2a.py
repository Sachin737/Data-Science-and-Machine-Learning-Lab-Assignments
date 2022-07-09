# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

import pandas as pd
import matplotlib.pyplot as plt

# Reading csv file
data = pd.read_csv("pima-indians-diabetes.csv")

# Creating list for all attributes
age = data["Age"].tolist()
preg = data["pregs"].tolist()
plas = data["plas"].tolist()
pres = data["pres"].tolist()
skin = data["skin"].tolist()
test = data["test"].tolist()
bmi = data["BMI"].tolist()
pedi = data["pedi"].tolist()


# Scatter plot of Age v/s pregs
plot1 = plt.figure(1)
plt.scatter(age, preg, marker='.')
plt.xlabel("Age", fontsize=8)
plt.ylabel("No of times pregnant", fontsize=8)
plt.show()

# Scatter plot of Age v/s plas
plot2 = plt.figure(2)
plt.scatter(age, plas, marker='.', color="g")
plt.xlabel("Age", fontsize=8)
plt.ylabel("Plasma conc.", fontsize=8)
plt.show()

# Scatter plot of Age v/s pres
plot3 = plt.figure(3)
plt.scatter(age, pres, marker='.', color="r")
plt.xlabel("Age", fontsize=8)
plt.ylabel("Diastolic blood pressure (mm Hg)", fontsize=8)
plt.show()

# Scatter plot of Age v/s skin
plot4 = plt.figure(4)
plt.scatter(age, skin, marker='.', color="c")
plt.xlabel("Age", fontsize=8)
plt.ylabel("Triceps skin fold thickness (mm)", fontsize=8)
plt.show()

# Scatter plot of Age v/s test
plot5 = plt.figure(5)
plt.scatter(age, test, marker='.', color="m")
plt.xlabel("Age", fontsize=8)
plt.ylabel("2-Hour serum insulin (mu U/mL)", fontsize=8)
plt.show()

# Scatter plot of Age v/s BMI
plot6 = plt.figure(6)
plt.scatter(age, bmi, marker='.', color="k")
plt.xlabel("Age", fontsize=8)
plt.ylabel("Body Mass Index", fontsize=8)
plt.show()

# Scatter plot of Age v/s pedi
plot7 = plt.figure(7)
plt.scatter(age, pedi, marker='.', color="r")
plt.xlabel("Age", fontsize=8)
plt.ylabel("Diabetes pedigree function", fontsize=8)
plt.show()

# Scatter plot of Age v/s Age
plot8 = plt.figure(8)
plt.scatter(age, age, marker='.', color="b")
plt.xlabel("Age", fontsize=8)
plt.ylabel("Age", fontsize=8)
plt.show()
