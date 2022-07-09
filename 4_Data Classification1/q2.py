# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import csv
import pandas as pd
import numpy as np

train_data = pd.read_csv('SteelPlateFaults-train.csv')  # Importing train_data set
train_class = train_data.iloc[:, -1]  # Storing Class label for train data set
train_data = train_data.drop('Class', axis=1)  # Dropping Class attribute

test_data = pd.read_csv('SteelPlateFaults-test.csv')  # Importing test_data set
test_class = test_data.iloc[:,-1]  # Storing Class label for test data set
test_data = test_data.drop('Class', axis=1)  # Dropping Class attribute

# Normalizing Train and test data
normalized_train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())
normalized_test_data = (test_data - train_data.min()) / (train_data.max() - train_data.min())

# Storing Normalized train data and test data in csv files.
normalized_train_data.to_csv('SteelPlateFaults-train-Normalised.csv', index=False)
normalized_test_data.to_csv('SteelPlateFaults-test-normalised.csv', index=False)

# values of k for KNN
k = [1, 3, 5]
max_acc = 0  # Storing maximum accuracy
kn = -1  # value of k for which accuracy is max.

# Performing KNN Classification
for i in k:
    knn_obj = KNeighborsClassifier(n_neighbors=i)  # Creating KNN object
    knn_obj.fit(normalized_train_data, train_class)  # Fitting normalized train data
    predict = knn_obj.predict(normalized_test_data)  # Prediction for normalized test data

    conf_matrix = confusion_matrix(test_class, predict, labels=[0, 1])  # Confusion matrix
    accuracy = round(accuracy_score(test_class, predict) * 100, 3)  # Classification Accuracy
    if accuracy > max_acc:
        max_acc = accuracy  # Finding max accuracy
        kn = i

    print(f'for k ={i}')
    print('Classification Accuracy: ', accuracy, '%')
    print('Confusion matrix : \n\n', conf_matrix, '\n')

print(f'Maximum accuracy is for k={kn} : {max_acc}')