# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

from sklearn.metrics import confusion_matrix, accuracy_score
from numpy.linalg import det, inv
import pandas as pd
import numpy as np
import csv

# Reading csv file
df_train = pd.read_csv('SteelPlateFaults-train.csv')
df_test = pd.read_csv('SteelPlateFaults-test.csv')


# Function to find likelihood for each tuple with each class
def likelihood(x, inverse_cov, mean_matrix, prior_prob):
    dot_product = np.dot(np.dot((x - mean_matrix).transpose(), inverse_cov), x - mean_matrix)
    return np.log(prior_prob) + 0.5 * np.log(det(inverse_cov)) - 11.5 * np.log(2 * np.pi) - 0.5 * dot_product


# Removing attributes as they make the covariance matrix singular
delete_attributes = ["TypeOfSteel_A300", "TypeOfSteel_A400", "X_Minimum", "Y_Minimum"]
for i in delete_attributes:
    df_train.pop(i)
    df_test.pop(i)

# Storing class 1 and class 0 tuples from train data set
a = df_train.groupby('Class')
train_class0 = df_train[df_train['Class'] == 0].iloc[:, 0:-1]
train_class1 = df_train[df_train['Class'] == 1].iloc[:, 0:-1]

mean_vec_0 = train_class0.mean()  # Mean vector for class0
mean_vec_0.to_csv('mean-vec0.csv')
mean_vec_1 = train_class1.mean()  # Mean vector for class1
mean_vec_1.to_csv('mean-vec1.csv')

cov_matrix_0 = pd.DataFrame(train_class0.cov())  # Covariance matrix for class0
cov_matrix_0.round(3).to_csv('Covariance_matrix0.csv')
inverse_cov_matrix_0 = inv(cov_matrix_0)  # Inverse of cov matrix class0

cov_matrix_1 = pd.DataFrame(train_class1.cov())  # Covariance matrix for class1
cov_matrix_1.round(3).to_csv('Covariance_matrix1.csv')
inverse_cov_matrix_1 = inv(cov_matrix_1)  # Inverse of cov matrix class1

# Prediction by bayes classifier
prediction = []
for i in df_test.index:
    prob_0 = likelihood(df_test.iloc[i, :23].to_numpy(), inverse_cov_matrix_0, mean_vec_0, 273 / (509 + 273))  # probability for class0
    prob_1 = likelihood(df_test.iloc[i, :23].to_numpy(), inverse_cov_matrix_1, mean_vec_1, 509 / (509 + 273))  # probability for class1
    if prob_0 > prob_1:
        prediction.append(0)
    else:
        prediction.append(1)

print('Accuracy: ', round(accuracy_score(df_test['Class'], prediction) * 100, 3), '% \n')
print('Confusion matrix:')
print(confusion_matrix(df_test['Class'], prediction))
