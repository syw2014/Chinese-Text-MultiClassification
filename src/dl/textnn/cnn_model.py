#!/usr/bin/env python
# -*- encoding: UTF-8 -*-
# Author: Jerry.Shi
# Date: 2017-10-19 11:00:30

# Description:
#       A simple implementation of TextCNN model, architecture, input -> embedding layer -> Conv -> pooling layer -> softmax -> output

import tensorflow as tf

class TextCNN(object):
    """Text CNN model"""
    def __init__(self, config):
        """Basic model config, and build cnn model.
        Args:
            config: 
        """
        self.config = config

        # placeholder
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        # keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)
        # build cnn model
        #self.build_cnn()

        # embedding layer
        with tf.device('cpu:/0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([self.config.vocab_size, 
                self.config.embedding_dim], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.emedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        
        # create convolution + max_pooling layer for each filter size
        pooled_output = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # convolution
                filter_shape = [filter_size, self.config.embedding_dim, 1, self.config.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                conv = tf.nn.conv2d(
                        self.emedded_chars_expanded,
                        W,
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="conv")
                # apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # maxpooling 
                pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.config.seq_length - filter_size+1, 1,1],
                        kstrides=[1,1,1,1],
                        padding="VALID",
                        name="pool")
                pooled_output.append(pooled)
        # combine all the pooled features
        num_filters_total = self.config.num_filters * self.config.num_filters
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

        # scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, self.config.num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(.1), shape=[self.config.num_classes], name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predict_y = tf.argmax(self.scores, 1, name="prediction")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_loss * 0.0

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(self.predict_y, tf.argmax(self.input_y, 1))
            self.accuarcy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")
""" 
    def input_embedding(self):
        input sequence embedding.
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable('embedding',
                    tf.random_uniform[self.config.vocab_size, self.config.embedding_dim], -1.0, 1.0)
            _inputs = tf.nn.embedding_lookup(self.W, self.input_x)
            _inputs = tf.expand_dims(self._inputs, -1)
        return _inputs

    def build_cnn(self):
        Build cnn model
        embedding_input = self.input_embedding()

        # TODO, try 2-dimension conv
        # convoluation layer, here use one dimension conv
        with tf.name_scope("conv_layer"):
            conv = tf.layers.conv1d(
                    embedding_input,
                    self.config.num_filters,
                    self.config.kernel_size, name='conv')

            # max pooling
            pooled = tf.reduce_max(conv, reduction_indices=[1], name='pooled')

        with tf.name_scope('score'):
            # fully connect layer before dropout and non-liner
            # TODO, optimize, use w*x + b
            fc = tf.layers.dense(pooled, self.config.hidden_dim, name='fc')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # logits, TODO, use w*x + b
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='logits')
            # predict
            self.pred_y = tf.argmax(tf.nn.softmax(self.logits),1)
        
        # calculate losses
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optimize = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        # accuracy
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.pred_y)
            self.accuray = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
"""
