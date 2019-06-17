import numpy as np 
import matplotlib.pyplot as plt 

dataxy = np.loadtxt('ex1data2.txt',dtype=float,delimiter=',')
X = dataxy[:,0:2]
y = dataxy[:,2]
X = np.reshape(X,(-1,2))
y = np.reshape(y,(-1,1))

X[:,0] = (1/ max(X[:,0])) * X[:,0]
X[:,1] = (1/ max(X[:,1])) * X[:,1]
y[:,0] = (1/ max(y[:,0])) * y[:,0]
print(max(y[:,0]),min(y[:,0]))

weights = np.random.rand(3,1)
weights = np.reshape(weights,(-1,1))
# print(np.shape(weights))
X_tr = np.c_[np.ones((47,1)),X]
X_train = X_tr[0:20,:]
X_test = X_tr[20:46,:]
y_train = y[0:20,:]
# print(X_tr)

learning_rate = 0.001
n_iter = 1000000
m = len(X)

for i in range(n_iter):
    cost = 1/m * ((X_train.dot(weights))-y_train)
    weights = weights -( learning_rate * X_train.T.dot(X_train.dot(weights)-y_train) )
# print(np.argmax(cost))

# plt.scatter(X,y)
# plt.show()
y_p = X_test.dot(weights)

plt.plot(X_test[:,0],y_p,'r-')
plt.plot(X[:,0],y,'b.')
plt.show()