import numpy as np 
import matplotlib.pyplot as plt 

#using Gradient Descent
dataxy = np.loadtxt('LinearRegression\ex1data1.txt',delimiter=',')

# print(dataxy)
#print(np.size(dataxy,0))
# print (len(dataxy))

X = dataxy[:,0]
Y = dataxy[:,1]
# print(X,Y)
# print(np.shape(X))
Xr= np.reshape(X,(-1,1))
Yr= np.reshape(Y,(-1,1))
# print(np.shape(Xr))




X_tr= np.c_[np.ones((len(X),1)),Xr]
# print(X_tr)
X_train = X_tr[0:50,:]
X_test = X_tr[50:,:]
Y_train = Yr[0:50,:]
Y_test = Yr[50:,:]

plt.scatter(X_train[:,1],Y_train)
plt.show()

eta = 0.000001 #Must not be more than that or either thetas will deminish
n_iter = 2000000
m=len(Y_train)

theta = np.random.rand(2,1)
#print(np.size(X_train,0))

for iterations in range(n_iter):
    gradients =2/m * X_train.T.dot(X_train.dot(theta)-Y_train)
    theta = theta - eta*gradients
   
# print(theta)
np.save('thetaTXT',theta)
theta=np.load('thetaTXT.npy')
# print(theta)

cost = 1/46*( (X_test.dot(theta)-Y_test))
print(max(cost))
# X_tst = np.array([[0],[2]])
# X_tst = np.c_[np.ones((2,1)),X_tst]
Y_predict = X_test.dot(theta)

plt.plot(X_test[:,1],Y_predict,"r-")
plt.plot(Xr,Yr,"b.")
#plt.axes([0,2,0,15])
plt.show()