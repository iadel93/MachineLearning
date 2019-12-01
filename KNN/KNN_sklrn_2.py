# this is a more advanced implementation of KNN using Numpy 

# import neccesary libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#import dataset
data = pd.read_csv('iris.csv')

#print the species / target labels
print(data['species'].unique())

# Encode the labels into numbers so we be able to use into the algorithm
le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])

print(data['species'].unique())

#printing the correlation between the data feaures
print(data.corr())
# plot the data to the target
plt.scatter(data['petal_length'],data['petal_width'],c=data['species'])
plt.show()

# Split the data into features and targets
x= np.array(data)
x= x[:,:4]

y = np.c_[data['species']]

x_trn,x_tst,y_trn,y_tst = train_test_split(x,y,test_size = 0.2)

# import the sklearn KNN class
from sklearn.neighbors import KNeighborsClassifier
accl =[]

# start the training algoritgh for different number of K neighbors 1 to 10
for k in range(1,10):
    # initialise the classifier
    classifier = KNeighborsClassifier(n_neighbors=k)
    # training on data
    classifier.fit(x_trn,y_trn.reshape(-1,))

    true = 0
    r=0
    # start calculating the accurace for each number of K
    for sample in x_tst:
        result = classifier.predict(sample.reshape(1,-1))
        if result == y_tst[r]:
            true +=1
        r+=1
    print("Accuracy = ", true)
    accl.append(true)
    
# plot the accuracy resulted from each K number of neighbors
plt.plot(range(1,10),accl)
plt.show()