# this code does implement the logistic regression using tensorflow and numpy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv

# loading the data from iris.txt
dataxy = np.loadtxt('iris.txt',delimiter=',',dtype=float,skiprows=1)

# split the data to features and targets
X=dataxy[:,0:4]
y=dataxy[:,4]
y = y.reshape(-1,1)

print(np.shape(y))

X_train = X_test = X
y_train = y_test = y

# hyperparameters
learning_rate = 0.0001
num_epochs = 1500
display_step = 1

# for visualize purpose in tensorboard we use tf.name_scope
with tf.name_scope("Declaring_placeholder"):
    # X is placeholdre for iris features. We will feed data later on
    X = tf.placeholder(tf.float32, [None, 4])
    # y is placeholder for iris labels. We will feed data later on
    y = tf.placeholder(tf.float32, [None, 1])
    
with tf.name_scope("Declaring_variables"):
    # W is our weights. This will update during training time
    W = tf.Variable(tf.zeros([4, 3]))
    # b is our bias. This will also update during training time
    b = tf.Variable(tf.zeros([3]))
    
with tf.name_scope("Declaring_functions"):
    # our prediction function
    y_ = tf.nn.softmax(tf.add(tf.matmul(X, W), b))

with tf.name_scope("calculating_cost"):
    # calculating cost
    cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
with tf.name_scope("declaring_gradient_descent"):
    # optimizer
    # we use gradient descent for our optimizer 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
with tf.name_scope("starting_tensorflow_session"):
    with tf.Session() as sess:
        # initialize all variables
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            cost_in_each_epoch = 0
            # let's start training
            _, c = sess.run([optimizer, cost], feed_dict={X: X_train, y: y_train})
            cost_in_each_epoch += c
            # you can uncomment next two lines of code for printing cost when training
            #if (epoch+1) % display_step == 0:
                #print("Epoch: {}".format(epoch + 1), "cost={}".format(cost_in_each_epoch))
        
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        # Calculate accuracy for 3000 examples
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({X: X_test, y: y_test}))