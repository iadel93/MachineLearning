import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

data = pd.read_csv('iris.csv')
# print(data['species'].unique())


le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])
# print(data['species'].unique())

data['species'] = data['species'].replace(to_replace =2,value = 1)
# print(data['species'].unique())
# print(data)
print(data.corr())

data = shuffle(data)
# print(data.head)

X = np.array(data[['sepal_length','sepal_width','petal_length','petal_width']])
X = np.c_[np.ones((150,1)) , X]
print(X.shape)
Y = np.array(data['species'])
Y = Y.reshape(-1,1)
print(Y.shape)

x_tr,x_ts,y_tr,y_ts = train_test_split(X,Y,test_size = 0.2)

learningRate = 0.00001
noIter = 100000
noFeatures = X.shape[1]
noSamples = x_tr.shape[0]

theta = np.zeros((noFeatures,1))

def sig(z):
    return (1/(1+np.exp(-1*z)))

for i in range(noIter):

    H = sig(x_tr.dot(theta))
    J = (-1*y_tr * (np.log(H)))-((1-y_tr)* (np.log(1-H)))
    err = H - y_tr
    theta = theta - (learningRate * (x_tr.T.dot(err)))

print(max(J))

correct = 0
pred = sig(x_ts.dot(theta))
pred = np.round(pred)
for i in range(len(pred)):
    if pred[i] == y_ts[i]:
        correct += 1
print('Correct = ',correct,'out of ',len(pred)) 