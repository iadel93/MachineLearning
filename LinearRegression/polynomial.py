#import the TF library and the numpy
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

#define the learning rate and the number of training epochs
learning_rate = 0.01
training_epochs = 400

#generate a X feature for training
trX = np.linspace(-1,1,101)
X_node = np.ones((trX.shape))
#define the number of coefficents
numcoeffs = 7
trY_coeffs = [1,1,2,3,4,5,6]
trY = 0

#Generate the Y for training and apply the polynomial function up to th i,th degree
for i in range(numcoeffs):
        if (i>1):
                trY += trY_coeffs[i] * np.power(trX, i)
        else:
                trY += X_node * 1.5

#scatter and plot the data
plt.scatter(trX, trY)
plt.show()

#define place TF place hoders for X and Y
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#Define the model function
def model(X,w):
    terms = []
    for i in range(numcoeffs):
        term = tf.multiply(w[i], tf.pow(X,i))
        terms.append(term)
    return tf.add_n(terms)

#Defining the TF variables
w = tf.Variable([0.]*numcoeffs, name = "parameters")
y_model = model(X,w)

#define the cost function
cost = (tf.pow(Y-y_model,2))
#Gradient descent training optimization technique
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Start the TF session and initialize it
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#start training for no. of training_epochs
for epoch in range(training_epochs):
    for (x,y) in zip(trX,trY):
        sess.run(train_op, feed_dict={X: x,Y: y})

#get the weights 
w_val = sess.run(w)
print(w_val)

sess.close()

#plot the resultant model out of training 
plt.scatter(trX, trY)
trY2 = 0
for i in range(numcoeffs):
    trY2 += w_val[i] * np.power(trX, i)

plt.plot(trX, trY2, 'r')
plt.show()