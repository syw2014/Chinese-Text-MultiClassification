#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Author: Jerry.Shi
# Date: 2017-05-12 9:43:30

# Description:
#   A simple cnn for mnist classification implementation based on tensorflow.

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# step 1, load data, one_hot is encoding for 10 labels
data_dir = "/data/research/data/mnist/input_data/"
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# tensorflow session
sess = tf.InteractiveSession()

# step 2, def weight initialize function
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# step 3, bias for ReLU active function
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# step 4, convolution function
def conv2d(x, W):
    """
    x: Input training image data, shape= [Number, 28*28*1]
    W: weight matrix,shape = [filter size=m*n, channel_number, filter_number]
    return filter_number images
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding ='SAME')

# step 5, max pooling, 2x2 -> 1x1
def max_pool_2x2(x):
    """
    x: input image, size: filter_size
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# step 6, input data preparation
x = tf.placeholder(tf.float32, [None, 784])   # load data variable
y_ = tf.placeholder(tf.float32, [None, 10])   # predict result
x_image = tf.reshape(x, [-1, 28, 28, 1])      # convert one-dimension data to specific size image

# step 7, first convolution layer
# set filter size = 5x5
# channel_num = 1
# filter_num = 32, which means extract 32 feature map
W_conv1 = weight_variable([5,5,1,32])
# As in the previous step we generated 32 feature map, so we need the same size bias for conv1
b_conv1 = bias_variable([32])
# first conv layer result
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# first pooling layer result
h_pool1 = max_pool_2x2(h_conv1)

# step 8, second convolution layer
# filter_size = 5x5
# channel_num = 32
# filter_num = 64, which means extract 64 feature maps
W_conv2 = weight_variable([5,5, 32 ,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# step 9, full connect layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# step 10, dropout for over fitting
# to decide how many data will be keeped
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# step 11, softmax layer, calculate the output probability
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# step 12, define loss and optimizer, cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# step 13, evaluate module
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# step 14, main train
# initialize all variables
tf.global_variables_initializer().run()
# hyperparameters for this model define as bellow:
# keep_prob = 0.5, mini_batch_size = 20, epoch = 20000
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, trianing accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# step 15, model test
# test accuracy
print("test accuracy %g" % accuracy.eval(feed_dict= {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
