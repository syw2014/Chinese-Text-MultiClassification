#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
sys.path.append('../util/')
from resnet_util import *

inputs = tf.placeholder(tf.float32, [3,4,4,6])
x = np.random.randn(3, 4, 4, 6)

net = identity_block(inputs, 
        f=2, 
        filters=[2, 4, 6], 
        stage=1, 
        block='a')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    net = sess.run([net], feed_dict={inputs: x})
    print("net out=" + str(net[0][1][1][0]))
