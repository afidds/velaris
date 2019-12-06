# Mapper 

After creating a classification for generated data of 4,000 points w/ 20 features, 6 significant the data was then visualized through mapper by labels and coordinate projections. By testing each i-th coordinate, I was able to determine which features were more or less significant based on the coloring. Notebook saved as K-means.ipynb 

## My Process & Experimentation 

I used this as a learning experience as well. I've been doing independent research on machine learning and data mining and wanted to implement a few things I learned. I started by installing the packages needed 
```python
from sklearn.cluster import KMeans 
from sklearn.datasets import make_classification
```
I then generated random data, and saved X, and Y as numpy arrays.I created an algorithm that clustered the data into 4 clusters and colored each cluster 
```python
kmeans = KMeans(n_clusters=4).fit(X)
labels = kmeans.predict(X, y) 
centroids = kmeans.cluster_centers_
```
After this, I imported mapper and fit the data to scale. Then the dictionary graph was created as 
```python
graph = mapper.map(projected_data, X, 
                   clusterer=sklearn.cluster.KMeans(n_clusters=4).fit(X), 
                   cover=km.Cover(n_cubes=10, perc_overlap=0.3, limits=None, verbose=0), 
                   nerve=km.GraphNerve(min_intersection=1), precomputed=False, remove_duplicate_nodes=False, 
                   overlap_perc=None, nr_cubes=None)
```
Lastly the visualization of the graph was was created with y as the color labels and x as the projection coordinate and saved it as an html file.
```python
mapper.visualize(graph, path_html="classifier_x10.html",
                 title="classifiers",
                 #color_function=y
                 color_function=X[:,10] 
                )
```
The K-means clustering didn't yield spectacular results. In the repository K-True.webarchive represents the basis. K-1, K-2, K-4 and K-5 are the only features that showed any significance. And this is because K-means is for unsupervised data, and in this case our data is labeled. 

I then created a K-Nearest Neighbor classification 
```python
from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors=4) 
knn.fit(X, y)
y_pred = knn.predict(X)
clean_dataset_score=knn.score(X, y)
```

After mapping this data I was able to see a bit more significance in the features. These graphs had a greater difference in node values. KNN-2 you can see that the nodes become more purple and some more green, X=3,4,5 all had some significance. From X=6,...,19 the graphs were more monotone.The KNN classification led to a better accuracy of the features. And the mapper was able to visualize that.
