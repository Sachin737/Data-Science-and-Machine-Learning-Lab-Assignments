# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

import matplotlib.pyplot as plt
import pandas as pd

# Reading csv file
data = pd.read_csv("pima-indians-diabetes.csv")


# List containing all the column's name
attributes = ['pregs', 'plas', 'pres', 'skin', 'test', 'BMI', 'pedi', 'Age']

# List containing labels for y-axix
ylabels = ['pregs', 'plas conc.', 'pres(mm-Hg)', 'skin(mm)', 'test(mu U/mL)', 'BMI(kg/m^2)', 'pedi', 'Age(years)']

# Plotting boxplot for all attributes
for i in range(8):
    plt.boxplot(data[attributes[i]])
    plt.xlabel("all patients of Pima Indian heritage")
    plt.ylabel(ylabels[i])
    plt.show()
