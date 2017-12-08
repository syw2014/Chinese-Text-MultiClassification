#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""Identity block implementation of residual network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def identity_block(inputs, f, filters, stage, block):
    """A identity block implementation based on the paper, He.etc https://arxiv.org/abs/1512.03385.
    This implementation was contains three layers:
        - 1x1 Conv + BN + Relu
        - fxf Conv + BN + Relu, f is to be specified
        - 1x1 Conv + BN
    Args:
        inputs: A tensor of size [batch_size, height_in, width_in, channels]
        f: intege, specifying the shape of the middle CONV size in the plain network
        filters, A list of integers, defining the number of filters in the CONV layers, 
            **Note**, this parameter in the raw implementation of Resnet model was an integer,
            the first layer and second layers has the same filters, the last layers has 4 * filters.
        stage: the name of layers depends on their position in the network, 
        block: string, name the layers
    return:
        net: output of the identity block, shape=[batch_size, h_new, w_new, c_new]
    """
    # define name bases
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    shortcut = inputs
    f1, f2, f3 = filters

    # First 1x1 CONV layer in plain network
    inputs = tf.layers.conv2d(
            inputs,
            filters=f1,
            kernel_size=1,
            strides=1,
            padding='valid',
            data_format='channels_last',
            kernel_initializer=tf.variance_scaling_initializer(),
            name=conv_name_base + "2a")
    # batch normalization
    inputs = tf.layers.batch_normalization(
            inputs,
            axis=3,
            momentum=_BATCH_NORM_DECAY,
            epsilon=_BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=True,
            fused=True,
            name= bn_name_base + "2a")
    # relu
    inputs = tf.nn.relu(inputs, name='relu')

    # Second fxf Conv
    inputs = tf.layers.conv2d(
            inputs,
            filters=f2,
            kernel_size=f,
            strides=1,
            padding='same',
            data_format='channels_last',
            kernel_initializer=tf.variance_scaling_initializer(),
            name=conv_name_base + "2b")
    # batch normalization
    inputs = tf.layers.batch_normalization(
            inputs,
            axis=3,
            momentum=_BATCH_NORM_DECAY,
            epsilon=_BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=True,
            fused=True,
            name= bn_name_base + "2b")
    # relu
    inputs = tf.nn.relu(inputs, name='relu')

    
    # Third 1x1 CONV
    inputs = tf.layers.conv2d(
            inputs,
            filters=f3,
            kernel_size=1,
            strides=1,
            padding='valid',
            data_format='channels_last',
            kernel_initializer=tf.variance_scaling_initializer(),
            name=conv_name_base + "2c")
    # batch normalization
    inputs = tf.layers.batch_normalization(
            inputs,
            axis=3,
            momentum=_BATCH_NORM_DECAY,
            epsilon=_BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=True,
            fused=True,
            name= bn_name_base + "2c")

    net = inputs + shortcut
    # relu
    net = tf.nn.relu(net, name='relu')

    return net
