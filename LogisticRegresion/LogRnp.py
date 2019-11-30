# the following code does implement the logistic regression using nothing more than 
# the numpy libraries and implements the logistic function by our hand

#import necessary libraries
import numpy as np 
import matplotlib.pyplot as plt 

#import the data from the files .txt
dataxy = np.loadtxt('ex2data1.txt',dtype=float,delimiter=',')

# split the data into features and targets
X = dataxy[:,(0,1)]
y = dataxy[:,2]

# reshaping the target set to the corect shape
y = y.reshape(-1,1)
print(np.shape(y))


# getting the rows and columns of the data
[m,n] = np.shape(X)

# creating the bias column of ones and adding it to the training dataset
X_tr=np.c_[np.ones((m,1)),X]

# create random weights for initial
thetas = np.zeros((n+1,1))

# defining the sigmoid function to be used
def sigmoid(z):
    return (1/(1 + np.exp(-1*z)))

# defining the number of iterations and the alpha
n_iter = 700000
alpha = 0.001

# start the training loop
for i in range(n_iter):
        # defining the cost function and calculate cost
    J = (-1/m)  *(y.T.dot(np.log(sigmoid(X_tr.dot(thetas)))) + ((1-y).T.dot(np.log(1-sigmoid(X_tr.dot(thetas))))))
        # get the result of the prediction im temp variable
    temp = sigmoid (X_tr.dot(thetas) )
        # calculate the error
    err = temp - y
        # get the gradient 
    grad = (1 / m) * (X_tr.T.dot(err) )
        # calculate the new thetas
    thetas = thetas - ((alpha)*(grad))
# print the final cost and the gradient
print(J,grad)

# reshape the thetas
thetas = thetas.reshape(-1,1)

# predict the training set using the new thetas after training
y_pred = sigmoid(X_tr.dot(thetas) )

# saving the final weights after the training
np.savetxt('thetasLog.txt',thetas,delimiter=',')

# calculating the accuracy
correct =0

for i in range(len(y_pred)):
    temp = np.round(y_pred[i])
    if temp == y[i]:
        correct +=1
    
print("Accuracy = ",(correct/len(y_pred)*100))

# plotting the data against the prediction
plt.plot(X_tr[:,1],y_pred,'r-')
plt.show()