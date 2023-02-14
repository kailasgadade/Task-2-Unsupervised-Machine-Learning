# # Task 2: Prediction using unsupervised ML Problem
# # Author : Gadade Kailas Rayappa
# # Step 1: Importing the Dataset
# Importing relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
# load the data
data=pd.read_excel("C:\\Users\\Kailas\\OneDrive\\Desktop\\Data II.xlsx")
# To show the first five columns of data set
data.head()
# Removing the 'Id' column
data.drop('Id' , axis=1, inplace=True) 
data.head()
# # Step 2 : Exploring the data
# no of rows & columns
data.shape
# Summary of Statistics
data.describe()
# To check the if null values are Present
data.info()
data.Species.value_counts()
# # Step 3 : Using the Elow Method to find the Optimal numbers of clusters
# Find the optimal number of clusters for k means clustering
x=data.iloc[:,:-1].values
from sklearn.cluster import KMeans
WCSS=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',
                 max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1,11),WCSS)
plt.title('The ELOW Method')
plt.xlabel('WCSS')
plt.show()
# Apply means to the dataset
kmeans=KMeans(n_clusters=3,
             max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)
y_kmeans
# # Step 5 : Visualize the test set result
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],
           s=100,c='red',label='Iris-setosa')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],
           s=100,c='blue',label='Iris-versicolour')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],
           s=100,c='green',label='Iris-virginica')
# Plotting the centorids of the clusters
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
           s=100,c='yellow',label='Centroids')
plt.legend()
# # Data Visualization
data.corr()
plt.figure(figsize=(12,5))
sns.heatmap(data.corr(),annot=True,cmap='ocean')
plt.figure(figsize=(8,6))
sns.boxplot(x="Species",y="SepalLengthCm",data=data)
sns.pairplot(data.corr())
# # Thank You!
