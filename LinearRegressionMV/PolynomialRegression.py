#!/usr/bin/env python
# coding: utf-8

# In[119]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize,StandardScaler

data = np.loadtxt('ex1data2.txt',delimiter = ',')
# print(data)

df = pd.DataFrame(data = data,columns = ["x1","x2","y"])
# print(df.head())
# print(df.corr())

# plt.scatter(df['x1'].values,df['y'].values)
# plt.show()

theta = [0.5,0.5,0.5,0.5]
theta = np.array(theta)
theta = np.reshape(theta,(1,-1))
X = df.drop('y',axis = 1).values
Y = df['y'].values
Y = np.reshape(Y,(-1,1))
m = X.shape[0]

print(X.shape)
print(m)
# print(X)
X = np.c_[np.ones((m,1)),X]
X = np.c_[X,(X[:,1])**2]
# print(X)
n = X.shape[1]
print(n)
X = StandardScaler().fit_transform(X)
Y = StandardScaler().fit_transform(Y)
print(Y.shape)
print(X)


# In[120]:


from sklearn.model_selection import train_test_split


# In[121]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
print(x_train.shape)


# In[122]:


alpha = 0.001


# In[123]:


for i in range(200000): 
    H = x_train.dot(theta.T)
#     print(H.shape)
    J = (sum((H-y_train)**2))/(2*37)
    
    err= y_train-H
    theta = theta + (alpha)*(err.T.dot(x_train))


# In[124]:


print(J)


# In[125]:


# print(y_train[:,0])
plt.scatter(x_train[:,1],y_train)
plt.plot(x_train[:,1],H,'r-')
plt.show()


# In[150]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = df.drop('y',axis = 1).values
Y = df['y'].values
Y = np.reshape(Y,(-1,1))

poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X) #contains X , X^2
# print(X_poly)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, Y)
theta =  lin_reg.coef_

print(theta)


X_tst = X_poly
# X_tst = X_tst.reshape(1,X_poly.shape[1])
Y_predict = lin_reg.predict(X_tst)
# print(Y_predict)
plt.plot(X_tst[:,0], Y_predict, "r-")
plt.scatter(X_tst[:,0], Y)
# plt.axis([0, 2, 0, 15])
plt.show()


# In[151]:


J = (sum((Y_predict-Y)**2))/(2*47)
print(J)


# In[ ]:




