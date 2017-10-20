#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Date: 2017-10-19 14:33:30

class ModelConfig(object):
    """Model configure parameters"""
    def __init__(self):
        # word embedding dimension
        self.embedding_dim = 64 
        # number of category
        self.num_classes = 265
        # number of filters
        self.num_filters = 256
        # size of conv kernel
        self.kernel_size = 5
        # vocab size
        self.vocab_size = 217639
        # the number of unit in full connect layer
        self.hidden_dim = 128
        
        # dropout
        self.dropout_keep_prob = 0.8
        # learning rate
        self.learning_rate = 0.001
        # batch size
        self.batch_size = 128
        # number of epoch
        self.epochs = 10
        # how many iterations to print results
        self.print_per_batch = 200
        # sentences length
        self.seq_length = 600
