from sklearn.datasets import load_iris
import numpy
from sklearn.cluster import k_means
import random
import pandas

# data (as pandas dataframes)
iris = load_iris()
X = iris.data
y = iris.target

#print(iris.feature_names)
#print(iris.target_names)

def randCent(dataSet, k):
 n = numpy.shape(dataSet)[1]
 centroids = numpy.asmatrix(numpy.zeros((k,n)))
 for j in range(n):
    minJ = min(dataSet[:,j])
    rangeJ = float(max(dataSet[:,j]) - minJ)
    centroids[:,j] = minJ + rangeJ * random.randint(k,1)
 return centroids

randCent(iris.data,4)