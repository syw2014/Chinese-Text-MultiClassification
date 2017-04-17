#! /usr/bin python
# -*- coding: utf-8 -*-
# File: DataGenerationAndFitting.py
# Author: Jerry.shi
# Date: 2017-03-27 15:10

import tensorflow as tf
import numpy as np
import time
# 1.data construction based on numpy, 100 random data
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300 # add noise
# print(x_data)
# print(y_data)

# 2. construct a linear model
# intercept
b = tf.Variable(tf.zeros([1]))
# coefficient
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 3. Minimum variance
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 4. initialize
init = tf.global_variables_initializer()

# 5. start graph
sess = tf.Session()
sess.run(init)

# 6. plane fitting
start = time.clock()
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

end = time.clock()
print("tian time: %0.3fs", end - start)
