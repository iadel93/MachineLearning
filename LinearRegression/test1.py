from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split


   
dataxy = np.loadtxt('LinearRegression\ex1data1.txt',delimiter=',')
print(dataxy)
# print(np.size(dataxy,1))
# print (len(dataxy))

X = dataxy[:,0]
y = dataxy[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# X = 2 * np.random.rand(100,1)
# Y = 3 * X + np.random.randn(100, 1)


print(type(X_test))
plt.scatter(X,y)
#plt.show()
print(X.shape)
X.reshape(97,1)
print(np.size(X.reshape(97,1)))
lin_reg = LinearRegression()
lin_reg.fit(X.reshape(-1,1), y)
print(lin_reg.intercept_)
print(lin_reg.coef_)

X_tst = np.array([[0],[2]])
X_test = X_test[:2,]
Y_predict = lin_reg.predict(X_test.reshape(-1,1))
plt.plot(X_tst, Y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 20, 0, 15])
plt.show()