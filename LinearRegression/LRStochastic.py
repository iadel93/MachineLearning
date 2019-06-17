import numpy as np 
import matplotlib.pyplot as plt 


X = 2 * np.random.rand(100,1)
y = 3 * X + np.random.randn(100, 1)
m=100
X_b= np.c_[np.ones((100,1)),X]
n_epochs = 50
t0, t1 = 5, 50 # learning schedule hyperparameters
def learning_schedule(t):
    return t0 / (t + t1)
theta = np.random.randn(2,1) # random initialization

#start the epochs
for epoch in range(n_epochs):
    for i in range(m): #loop on every sample in training data set and apply the gradient descent
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

#print the final thetas after the training
print(theta)