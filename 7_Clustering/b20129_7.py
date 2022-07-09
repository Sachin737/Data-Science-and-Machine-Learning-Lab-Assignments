# Name : Sachin Mahawar
# Roll no. : b20129
# Mobile no. : 9166843951

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN


def purity_score(real, pred):  # function to calculate purity score
    contingency_matrix = metrics.cluster.contingency_matrix(real, pred)  # compute contingency matrix (also called confusion matrix)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)  # Find optimal one-to-one mapping between cluster labels and true labels
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)  # Return cluster accuracy


df = pd.read_csv('Iris.csv')  # importing csv file
attributes = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
clrs = ['green', 'blue', 'orange','red']
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# disable chained assignments
pd.options.mode.chained_assignment = None

# Replacing Species with numerical values
for i in range(0, df.shape[0]):
    if df['Species'][i] == 'Iris-setosa':
        df['Species'][i] = '0'
    elif df['Species'][i] == 'Iris-versicolor':
        df['Species'][i] = '1'
    else:
        df['Species'][i] = '2'

new_df = df.iloc[:, :-1]  # Removing "Species" class

# Q1

# Performing PCA analysis
# reducing data from 4 dimensional to 2 dimensional

p = PCA(n_components=2)
reduced_data = p.fit_transform(new_df)

v = [1,2,3,4]
cor = df.corr().to_numpy()
eig_val,eigen_vec = np.linalg.eig(cor)

plt.figure(1)
plt.plot(v,eig_val, label="eigen values")
plt.xlabel("Components")
plt.ylabel("Eigen values")
plt.title("Eigen Values vs Components")
plt.show()

# Q2
print('Q2.')
# K-means clustering
kmeans = KMeans(n_clusters=3)  # Creating Kmeans object
kmeans.fit(reduced_data)  # Training it with data
kmeans_prediction = kmeans.predict(reduced_data)  # Prediction
centres = kmeans.cluster_centers_  # Finding clusters centre

for i in range(3):  # Iterating over 3 clusters
    # Scatter plot
    plt.figure(2)
    plt.scatter(reduced_data[kmeans_prediction == i, 0], reduced_data[kmeans_prediction == i, 1], color=clrs[i], label=labels[i])
    plt.legend()
    plt.scatter(centres[i, 0], centres[i, 1], marker='X', color='red')  # Plotting centres
    plt.xlabel("Dimension 1"), plt.ylabel("Dimension 2")
    plt.show()

print('distortion measure : ', kmeans.inertia_)  # Printing Distortion
print("Purity score for test example : ", round(purity_score(df.iloc[:, -1], kmeans_prediction), 3))  # Printing purity score


# Q3
print('Q3.')

K = [2, 3, 4, 5, 6, 7]  # Values of K
distortion = []
purity_scores = []

for k in K:
    kmeans = KMeans(n_clusters=k)  # Creating K-means object
    kmeans.fit(reduced_data)  # Training it with data
    kmeans_prediction = kmeans.predict(reduced_data)  # Prediction

    # Appending distortion, purity score
    distortion.append(round(kmeans.inertia_, 3))
    purity_scores.append(round(purity_score(df.iloc[:, -1], kmeans_prediction), 3))

# Plotting K vs distortion measure
plt.figure(3)
plt.plot(K, distortion)
plt.title('Distortion vs K')
plt.xlabel('K-value');plt.ylabel('Distortion')
plt.show()

# Optimal cluster:
# By elbow method, we can say that optimal value of k is 3
# for which distortion is 63.874 and purity score is 0.887

print('The distortion measures are', distortion)
print('The purity scores are', purity_scores)

# Q4
print('Q4.')

gmm = GaussianMixture(n_components=3,random_state=42).fit(reduced_data)  # Creating gmm object & fitting it with data
gmm_pred = gmm.predict(reduced_data)    # Predictions
gmm_centres = gmm.means_    # Centres

# (a)
# Plotting the scatter plot
plt.figure(4)
plt.scatter(reduced_data[:,0],reduced_data[:,1], c=gmm_pred, cmap='rainbow', s=15)  # Scatter plot for data points showing each cluster
plt.scatter( centres[:,0],centres[:,1],label='cluster centres')     # Centres of cluster
plt.legend()
plt.title('GMM')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# (b)
print('The total data log likelihood :', round(gmm.score(reduced_data) * len(reduced_data), 3))
# (c)
print('The purity score for k =3 :', round(purity_score(gmm_pred,df.iloc[:,-1]), 3))

# Q5
print('Q5.')

total_log_likelihood = []   # Storing log likelihood (distortion measure)
purity = []     # Storing purity

for i in K:
    gmm = GaussianMixture(n_components=i, random_state=42).fit(reduced_data)    # Creating gmm object & fitting it with data
    total_log_likelihood.append(round((gmm.score(reduced_data) * len(reduced_data)), 3))    # Appending total log likelihood
    purity.append(round(purity_score(gmm.predict(reduced_data),df.iloc[:,-1]), 3))  # Appending purity

print('The distortion measures :', total_log_likelihood)
print('The purity scores :', purity)

plt.figure(5)
plt.plot(K, total_log_likelihood)
plt.title('K vs distortion measure')
plt.xlabel('K')
plt.ylabel('Distortion Measure')
plt.show()

# Q6
print('Q6.')

min_samples = [4,10]
epsilon = [1,5]
z = 6   # to get different plots

for i in range(2):
    for j in range(2):
        dbscan_model = DBSCAN(eps=epsilon[i], min_samples=min_samples[j]).fit(reduced_data)  # Creating dbscan model
        DBSCAN_predictions = dbscan_model.labels_   # Predictions

        plt.figure(z)
        # Scatter plot
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=DBSCAN_predictions, cmap='flag', s=15)
        plt.title(f'eps={epsilon[i]} and min_samples={min_samples[j]} Data points')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
        z+=1

        # Printing purity score for each combination of eps and min_samples
        print(f'Purity score for eps={epsilon[i]} ans min_sample={min_samples[j]} :',
              round(purity_score(DBSCAN_predictions, df.iloc[:,-1]),3))







