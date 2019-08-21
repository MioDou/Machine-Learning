# -*- coding: utf-8 -*-
# Lab 9 XOR
import tensorflow as tf
import numpy as np

tf.compat.v1.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

tf.compat.v1.disable_eager_execution()

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.compat.v1.placeholder(tf.float32, [None, 2])
Y = tf.compat.v1.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random.normal([2, 2]), name='weight1')
b1 = tf.Variable(tf.random.normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random.normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random.normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.math.log(hypothesis) + (1 - Y) *
                       tf.math.log(1 - hypothesis))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.compat.v1.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, '   ',sess.run(cost, feed_dict={X: x_data, Y: y_data}), '   ',sess.run([W1, W2]))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)


'''
Hypothesis:  [[ 0.01338218]
 [ 0.98166394]
 [ 0.98809403]
 [ 0.01135799]]
Correct:  [[ 0.]
 [ 1.]
 [ 1.]
 [ 0.]]
Accuracy:  1.0



Hypothesis:  [[0.01231546]
 [0.98866475]
 [0.9836283 ]
 [0.01044083]] 
Correct:  [[0.]
 [1.]
 [1.]
 [0.]] 
Accuracy:  1.0


Hypothesis:  [[0.01193672]
 [0.9836838 ]
 [0.98365986]
 [0.015297  ]] 
Correct:  [[0.]
 [1.]
 [1.]
 [0.]] 
Accuracy:  1.0

'''
