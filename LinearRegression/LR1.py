#importing the essential libraries
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

#initialising Learning Rate and training eterations
learning_rate = 0.01  
training_epochs = 100

Xtrain= np.linspace(-1, 1, 101)  #geneating a linearly seperable data from -1 to 1
Xtrain = np.c_[np.ones((*Xtrain.shape )),Xtrain]  #adding the intercept X-node
Ytrain= 2*Xtrain + np.random.randn(*Xtrain.shape ) * 0.33  #generating Y upon the X features

# plt.scatter(Xtrain,Ytrain)
# plt.show()

#Adding two TF placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#Defining the model function
def model(X, w ):
    return tf.multiply(X,w)

#Defining the weight function
w = tf.Variable(0.0, name="weights")

y_model = model(X,w)
cost = tf.square(Y - y_model) #the cost function

#TF training object as GradientDescent
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#opening a TF session to start training
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#Training
for epochs in range(10):
    for(x,y) in zip(Xtrain,Ytrain):
        sess.run(train_op,feed_dict={X:x,Y:y})

#Retrieve the weights
w_val = sess.run(w)
print(w_val)
sess.close() #close the TF session

#plotting the result
plt.scatter(Xtrain,Ytrain)
y_learned= Xtrain*w_val
plt.plot(Xtrain,y_learned,'r-')
plt.show()
