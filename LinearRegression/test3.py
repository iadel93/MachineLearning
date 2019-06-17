import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = np.loadtxt('LinearRegression\ex1data2.txt',dtype=float,delimiter=',')
# print(data)

X= data[:,:2]
y= data[:,2]
max1 = max(X[:,0])
max2 = max(X[:,1])
X[:,0]= X[:,0]/max1
X[:,1]= X[:,1]/max2
        
print(X)
# print(X.shape)
y = np.reshape(y,(-1,1))
# print(y.shape)

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2)

print(X_train.shape)

# plt.scatter(X_train[:,0],y_train)
# plt.show()

lr = LinearRegression()
lr.fit(X_train,y_train,sample_weight=0.05)
print(lr.coef_)
print(lr.intercept_)

y_pred = lr.predict(X_test)

error = 1/9*np.sum((y_pred-y_test))
print(error)

plt.plot(X_test, y_pred, "r-")
plt.plot(X_train, y_train, "b.")
# plt.axis([0, 2, 0, 15])
plt.show()