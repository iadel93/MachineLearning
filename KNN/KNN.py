# in this sourcecode we implement the KNN classifier using only the numpy
# and then implement the algorithm ourselves

# Importing libraries
import pandas as pd
import numpy as np
import math
import operator

#### Start of STEP 1
# Importing data 
data = pd.read_csv("iris.csv")
#### End of STEP 1 

# Defining a function which calculates euclidean distance between two data points
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)

# Defining our KNN model
def knn(trainingSet, testInstance, k):
 
    distances = {}
    sort = {}
 
    length = testInstance.shape[1]
    
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)
        distances[x] = dist[0]
 
    # Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1)) 
    neighbors = []
    
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    classVotes = {}
    
    # Calculating the most frequent class in the neighbors
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
 
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedVotes[0][0], neighbors)

# Creating a dummy testset
testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)

# Setting number of neighbors = 1
k = 1

# Running KNN model
result,neigh = knn(data, test, k)

# Predicted class
print(result)


