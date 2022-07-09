# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

import pandas as pd
import matplotlib.pyplot as plt

# Reading csv file
data = pd.read_csv("pima-indians-diabetes.csv")

# Creating list for all attributes
bmi = data["BMI"].tolist()
preg = data["pregs"].tolist()
plas = data["plas"].tolist()
pres = data["pres"].tolist()
skin = data["skin"].tolist()
test = data["test"].tolist()
pedi = data["pedi"].tolist()
age = data["Age"].tolist()


# Scatter plot of BMI v/s pregs
plot1 = plt.figure(1)
plt.scatter(bmi, preg, marker='.')
plt.xlabel("Body Mass Index", fontsize=8)
plt.ylabel("No of times pregnant", fontsize=8)
plt.show()

# Scatter plot of BMI v/s plas
plot2 = plt.figure(2)
plt.scatter(bmi, plas, marker='.', color="g")
plt.xlabel("Body Mass Index", fontsize=8)
plt.ylabel("Plasma conc.", fontsize=8)
plt.show()

# Scatter plot of BMI v/s pres
plot3 = plt.figure(3)
plt.scatter(bmi, pres, marker='.', color="r")
plt.xlabel("Body Mass Index", fontsize=8)
plt.ylabel("Diastolic blood pressure (mm Hg)", fontsize=8)
plt.show()

# Scatter plot of BMI v/s skin
plot4 = plt.figure(4)
plt.scatter(bmi, skin, marker='.', color="c")
plt.xlabel("Body Mass Index", fontsize=8)
plt.ylabel("Triceps skin fold thickness (mm)", fontsize=8)
plt.show()

# Scatter plot of BMI v/s test
plot5 = plt.figure(5)
plt.scatter(bmi, test, marker='.', color="m")
plt.xlabel("Body Mass Index", fontsize=8)
plt.ylabel("2-Hour serum insulin (mu U/mL)", fontsize=8)
plt.show()

# Scatter plot of BMI v/s pedi
plot6 = plt.figure(6)
plt.scatter(bmi, pedi, marker='.', color="r")
plt.xlabel("Body Mass Index", fontsize=8)
plt.ylabel("Diabetes pedigree function", fontsize=8)
plt.show()

# Scatter plot of BMI v/s age
plot7 = plt.figure(7)
plt.scatter(bmi, age, marker='.', color="k")
plt.xlabel("Body Mass Index", fontsize=8)
plt.ylabel("Age", fontsize=8)
plt.show()
