import numpy as np 
import matplotlib.pyplot as plt

data  = [[1,4],[2,8],[3,12],[4,16],[5,20]]
data = np.array(data)
print(data)

X = data[:,0]
Y = data[:,1]
X = X.reshape((-1,1))
Y = Y.reshape((-1,1))
print(np.shape(X))
print(np.shape(Y))

plt.scatter(X,Y)
# plt.show()

theta = np.array([0.5,0.5])
theta = theta.reshape((-1,1))
print(np.shape(theta))

alpha = 0.01
X_node = np.ones((5,1))
X = np.c_[X_node,X]

print(np.shape(X))
for i in range(1000):
    H = X.dot(theta)
    # print(H)
    err = H-Y
    J = (0.1)* (np.sum(err)**2)
    # print(J)
    theta = theta + ((alpha)*X.T.dot((Y-H)))
print(theta)
print(round(J))
H = X.dot(theta)
plt.plot(X[:,1],H,'r-')
plt.show()