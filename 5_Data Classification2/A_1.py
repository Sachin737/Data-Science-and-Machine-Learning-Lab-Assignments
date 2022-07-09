# Name : Sachin Mahawar
# Roll number : B20129
# Mobile Number : 9166843951

# Importing libraries
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix,accuracy_score
import pandas as pd

# Reading csv files
df_train = pd.read_csv('SteelPlateFaults-train.csv')
df_test = pd.read_csv('SteelPlateFaults-test.csv')

# Removing these attributes as they make the covariance matrix singular
delete_attributes = ["TypeOfSteel_A300", "TypeOfSteel_A400", "X_Minimum", "Y_Minimum"]
for i in delete_attributes:
    df_train.pop(i)
    df_test.pop(i)

# Storing class 1 and class 0 tuples from train data set
train_class0 = df_train[df_train['Class'] == 0].iloc[:, 0:-1]
train_class1 = df_train[df_train['Class'] == 1].iloc[:, 0:-1]

test_label =df_test.pop('Class')  # Popping Class attribute
df_train.pop("Class")

# Computing prior probability of 2 classes
prior0 = train_class0.shape[0] / (train_class0.shape[0] + train_class1.shape[0])
prior1 = train_class1.shape[0] / (train_class0.shape[0] + train_class1.shape[0])

# Building GMM for different values of q
Q = [2, 4, 8, 16]

for q in Q:
    # fitting the model for class0 and class1 for Q=q
    GMM_class0 = GaussianMixture(n_components=q, covariance_type='full', reg_covar=1e-4)
    GMM_class0.fit(train_class0)
    GMM_class1 = GaussianMixture(n_components=q, covariance_type='full', reg_covar=1e-4)
    GMM_class1.fit(train_class1)

    prediction = []     # List to store the predicted class of test sample
    a0 = GMM_class0.score_samples(df_test)  # Computing likelihood probability for class0
    a1 = GMM_class1.score_samples(df_test)  # Computing likelihood probability for class1

    for i in range(len(a0)):
        if np.log(prior0)+a0[i] > np.log(prior1)+a1[i]:  # If P(X/C0)*P(C0)> P(X/C1)*P(C1) means sample is of 0 class as per Bayes classifier
            prediction.append(0)
        elif np.log(prior0)+a0[i] < np.log(prior1)+a1[i]:
            prediction.append(1)

    confusion_mat = confusion_matrix(test_label,prediction)  # Confusion matrix
    accuracy = accuracy_score(test_label,prediction)    # Compute accuracy
    print(f'For Q = {q}')
    print("Confusion matrix : \n",confusion_mat)
    print("Accuracy :",accuracy,'\n')
