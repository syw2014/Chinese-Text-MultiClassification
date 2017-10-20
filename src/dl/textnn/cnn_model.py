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

        # build cnn model
        self.build_cnn()

    def input_embedding(self):
        """input sequence embedding."""
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding = tf.get_variable('embedding',
                    [self.config.vocab_size, self.config.embedding_dim])
            _inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        return _inputs

    def build_cnn(self):
        """Build cnn model"""
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
            


