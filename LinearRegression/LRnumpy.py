import numpy as np 
import matplotlib.pyplot as plt
 
#using normal Equation
#initilaise X and Y as random numbers
X = 2 * np.random.rand(100,1)
Y = 3 * X + np.random.randn(100, 1)

#plotting the data
plt.scatter(X,Y)
plt.show()

#adding the intercept to X
X_tr= np.c_[np.ones((100,1)),X]
#Applying the equation method to get the best theta
theta_best = np.linalg.inv(X_tr.T.dot(X_tr)).dot(X_tr.T).dot(Y)

print(theta_best)

#testing the model on a new dataset
X_tst = np.array([[0],[2]])
X_tst_full = np.c_[np.ones((2,1)),X_tst]
Y_predict = X_tst_full.dot(theta_best)

print(Y_predict)

#plot the model

plt.plot(X_tst, Y_predict, "r-")
plt.plot(X, Y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

#