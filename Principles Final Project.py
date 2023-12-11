'''
Author: Jack Bielawa
Professor: Dr. Rivera
Class: Comsc 230
Assignment: Final Project
Date: 12/8
Program Name: Iris Testing
Program Description: Provides desciptive statistics of the IRIS dataset and then clusters and plots the data as well as 
                     uses k-nearest-neighbor to create a predictive model.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA



iris = datasets.load_iris()
X = iris.data
y = iris.target

# clusters the data into 3 groups using kmeans
kmeans = KMeans(n_clusters=3, random_state=50)
kmeans.fit(X)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# transforms 4 variables for each flower into two for graphing purposes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the data
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=150, linewidths=3, color='red', label='Centroids')
plt.title('K-means Clustering of Iris Dataset')
#plt.xlabel('')
#plt.ylabel('')
#plt.legend()
plt.show()



from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings

# Filter the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.neighbors._classification")


#read data from pc
df = pd.read_csv("IRIS.csv")

#gets overall descriptive statistics
print(df.describe())
print("\n\n")

#takes averages of lengths and widths for each flower species
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target_names[iris.target]
average_lengths = data.groupby('species').mean()
print(average_lengths[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])


print("\n\nAccuracy: \n")




#train model using .70 for training and .30 for testing
X_train, X_test, y_train, y_test = train_test_split(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], df['species'], test_size = .3)
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)


#print correct / amount tested
correct = np.where(prediction == y_test, 1, 0).sum()
print(correct, "/", len(y_test))

#print accuracy of model
accuracy = correct/len(y_test)
print(accuracy)


