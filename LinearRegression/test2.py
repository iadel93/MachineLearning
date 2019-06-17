import numpy as np 
import matplotlib.pyplot as plt

data = np.loadtxt("LinearRegression\ex1data2.txt",dtype=float,delimiter=",")

# print(data)
# print(np.size(data,1))
feature_no =np.size(data,1)-1 

X=data[:,0:feature_no]
# print(X)
Y=data[:,feature_no]
X= np.reshape(X,(-1,feature_no))
Y= np.reshape(Y,(-1,1))
max1 = max(X[:,0])
max2 =max(X[:,1])

X[:,0] = X[:,0]/max1 
X[:,1] = X[:,1]/max2
X[:,1] = X[:,1]**2
# print(X)

# print(np.shape(X))
# print(np.shape(Y))

m = np.size(X,0)
alpha = 0.1
# theta = np.random.randn(2,1)
theta = 0.5 * np.ones((feature_no+1,1))

# print(theta)
# print(len(X))
xnode = np.ones((len(X),1))
X = np.c_[xnode,X]
# print(X)
plt.scatter(X[:,1],Y)
plt.show()

for i in range(10000):
    H = np.dot(X,theta)
    error = H-Y
    J = (1/(2*len(X)))*(np.sum(error))**2
    Xtranspose = np.transpose(X)
    theta = theta - (alpha/m) * np.transpose(error.T.dot(X))

# print(len(H))
print(J)
Y_predict = X.dot(theta)
# print(np.size(theta,1))
plt.plot(X[:,1],Y_predict,"r--")
plt.plot(X[:,1],Y,"b.")
#plt.axes([0,2,0,15])
plt.show()