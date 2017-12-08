#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Date: 2017-10-19 14:33:30

class ModelConfig(object):
    """Model configure parameters"""
    def __init__(self):
        # word embedding dimension
        self.embedding_dim = 128 
        # number of category
        self.num_classes = 265
        # number of filters
        self.num_filters = 128
        # comma-separated filter sizes
        self.filter_sizes = "3,4,5"
        # vocab size
        self.vocab_size = 217639
        # the number of unit in full connect layer
        self.hidden_dim = 128
        
        # dropout
        self.dropout_keep_prob = 0.5
        # learning rate
        self.learning_rate = 1e-3
        # batch size
        self.batch_size = 64
        # number of epoch
        self.epochs = 10
        # sentences length
        self.seq_length = 600
    
        # training parameters
        # Evaluate model on dev set after this many steps, default 100
        self.evaluate_every = 100

