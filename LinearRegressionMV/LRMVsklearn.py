import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# define the number of iterations
m = 100
# define feature X1
X = 6 * np.random.rand(m, 1) - 3
#generate target Y as a function of X1
Y = X**3+0.5 * X**2 + X * 2 + np.random.randn(m, 1)
# plot the data as opposed to X1 and target Y
plt.scatter(X,Y)
plt.show()

# Transform the feature X1 into higher degree (Polynomial) Feature of degree 3
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X) #contains X , X^2

# Initialize the Linear regression model in the SKLEARN 
lin_reg = LinearRegression()
lin_reg.fit(X_poly, Y) # Start fitting / Training 
theta =  lin_reg.coef_ # Retrieve the Thetas

print(theta)

# Test the model
X_tst = X_poly[5,:]
Y_predict = lin_reg.predict(X_tst.reshape(1,-1))

# Plot the results
plt.plot(X_tst[:,], Y_predict, "r-")
plt.plot(X, Y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()