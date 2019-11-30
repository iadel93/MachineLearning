# the following code does implement the logistic regression using sci-kit learn 
# library and shows graph of the training result (accuracy)

#import libraries needed
from sklearn  import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np 
import matplotlib.pyplot as plt 

# load the iris dataset from sklearn
iris = datasets.load_iris()

# print the species of the dataset
print(list(iris.keys()))

# split the data to features and targets
X = iris["data"][:,3:]
y = (iris["target"]==2).astype(np.int)

#print the features and targets
print(X, y)

#create a logistic regression class 
log_reg = LogisticRegression()

#start training the model
log_reg.fit(X,y)

# create random features inorder to be used for testing 
X_new = np.linspace(0,3,1000).reshape(-1,1)

#predicting the data generated
y_pred = log_reg.predict_proba(X_new)

#plotting the features against the prediction where it shows a definite classification area
plt.plot(X_new, y_pred[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_pred[:, 0], "b--", label="Not Iris-Virginica")
plt.show()

