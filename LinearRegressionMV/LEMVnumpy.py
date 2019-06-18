import numpy as np
import matplotlib.pyplot as plt 
# Define the number of samples
m=100

#Start to generate random feature X1
X1 = 6 * np.random.rand(m, 1) - 3
# Start to generate random feature X2
X2 = 3 * np.random.rand(m, 1) - 3
# Add X1,X2 to X0 which is the intercept feature
X = np.c_[np.ones((100,1)),X1,X2]
# Generate Y as a function of X1 and X2
Y = 0.5 * X2 + X1 * 2 + np.random.randn(m, 1)
# Plot the data generated as opppsed to feature X1  and target Y
plt.scatter(X1,Y)
plt.show()
# print(X)

eta = 0.1  #define the learning rate
n_iter = 100  #define the number of iterations
theta = np.zeros((3,1))  #initialize theta as zeros

# Start the training iterations
for i in range(n_iter):
    H = X.dot(theta)
    error = H - Y
    gradient = (1/m)* (eta)*(X.T.dot(error))  # Calculate the Gradient
    theta = theta - gradient

# Print the final thetas that are deduced
print (theta)
