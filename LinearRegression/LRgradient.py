#importing the related libraries
import numpy as np 
import matplotlib.pyplot as plt 

#using Gradient Descent
#initialising the X , and Y as random numbers
X = 2 * np.random.rand(100,1)
Y = 3 * X + np.random.randn(100, 1)

#plotting the data
plt.scatter(X,Y)
plt.show()

#adding the intercept
X_tr= np.c_[np.ones((100,1)),X]

#defining the initial weights and the number of iterations
eta = 0.1
n_iter = 1000
m=100

theta = np.random.randn(2,1)

#Start applying the gradient descent (Training)
for iterations in range(n_iter):
    gradients = 2/m * X_tr.T.dot(X_tr.dot(theta)-Y)
    theta = theta - eta*gradients

print(theta)

print(np.size(X_tr,1))
X_tst = np.array([[0],[2]])
X_tst = np.c_[np.ones((2,1)),X_tst]
Y_predict = X_tst.dot(theta) #Predicting the results of a test dataset
#plotting the model and see how does it fit
plt.plot(X_tst[:,1],Y_predict,"r-")
plt.plot(X,Y,"b.")
plt.axes([0,2,0,15])
plt.show()