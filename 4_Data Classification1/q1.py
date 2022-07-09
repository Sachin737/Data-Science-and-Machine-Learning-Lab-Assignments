# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import csv
import pandas as pd
import numpy as np

# Reading csv file
df = pd.read_csv('SteelPlateFaults-2class.csv')

# Splitting data into 30-70% for train and test data form each class
train_data1, test_data1, train_class1, test_class1 = train_test_split(df[df['Class'] == 0].iloc[:, :-1], df[df['Class'] == 0].iloc[:, -1], test_size=0.3, random_state=42, shuffle=True)
train_data2, test_data2, train_class2, test_class2 = train_test_split(df[df['Class'] == 1].iloc[:, :-1], df[df['Class'] == 1].iloc[:, -1], test_size=0.3, random_state=42, shuffle=True)

data1 = train_data1.merge(train_class1, left_index=True, right_index=True)  # Merging train data from class 0 with its label/class
data2 = train_data2.merge(train_class2, left_index=True, right_index=True)  # Merging train data from class 1 with its label/class
train_data_final = pd.concat([data1, data2])  # concatenating train data of  both the classes
train_data_final.to_csv("SteelPlateFaults-train.csv", index=False)  # Storing train data set in csv file

data3 = test_data1.merge(test_class1, left_index=True, right_index=True)  # Merging test data from class 0 with its label/class
data4 = test_data2.merge(test_class2, left_index=True, right_index=True)  # Merging test data from class 1 with its label/class
test_data_final = pd.concat([data3, data4])  # concatenating train data of  both the classes
test_data_final.to_csv('SteelPlateFaults-test.csv', index=False)  # Storing test data set in csv file

# values of k for KNN
k = [1, 3, 5]
max_acc = 0  # Storing maximum accuracy
kn = -1  # value of k for which accuracy is max.

# Performing KNN Classification
for i in k:
    knn_obj = KNeighborsClassifier(n_neighbors=i)  # Creating KNN object
    knn_obj.fit(train_data_final.iloc[:, :-1], train_data_final.iloc[:, -1])  # Fitting train data
    predict = knn_obj.predict(test_data_final.iloc[:, :-1])  # Prediction for test data

    conf_matrix = confusion_matrix(test_data_final.iloc[:, -1], predict, labels=[0, 1])  # Confusion matrix
    accuracy = round(accuracy_score(test_data_final.iloc[:, -1], predict) * 100, 3)  # Classification Accuracy
    if accuracy > max_acc :
        max_acc = accuracy  # Finding max accuracy
        kn = i

    print(f'for k ={i}')
    print('Classification Accuracy: ', accuracy, '%')
    print('Confusion matrix : \n', conf_matrix, '\n')

print(f'Maximum accuracy is for k={kn} : {max_acc}')
