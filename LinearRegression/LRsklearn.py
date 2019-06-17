#import the important libraries for plotting and scikit-learn
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt 

#initialise the X and Y with random numbers
X = 2 * np.random.rand(100,1)
Y = 3 * X + np.random.randn(100, 1)

#plot the data
plt.scatter(X,Y)
plt.show()

#Using the Linear regression model from sklearn
lin_reg = LinearRegression()
#Try fitting the data directly (training)
lin_reg.fit(X, Y)
print(lin_reg.intercept_)

print(lin_reg.coef_)#Get the model weights after training

#generate a new test data
X_tst = np.array([[0],[2]])
print(len(X_tst))
Y_predict = lin_reg.predict(X_tst)
print(len(X_tst))

#plot the result
plt.plot(X_tst, Y_predict, "r-")
plt.plot(X, Y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()