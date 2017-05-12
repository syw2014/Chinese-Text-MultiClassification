#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Author: Jerry.Shi
# Date: 2017-05-12 18:16:30
# Description:
#       An util tool for load text data and convert to libsvm format.Input is the data directory.
#   Process as bellow:
#       step1, load text data, store label list(var: total_labels=[]), aggregate the same 
#           lable as one list(var: label_samples={})
#       step2, extract keywords for every label based on term frequency after remove stopwords, (var: label_keywords={label: []})
#       step3, generate a global feature words based on IDF in label keywords,(var: feature_weight={}), in this case we only consider
#           IDF, later may use TFIDF or other model to complete optimization
#       step4, use global feature vectorize all the samples, finally return label and text vector.

from __future__ import division
from __future__ import print_function

import numpy as np
import math, scipy, time
from tqdm import *

class DataUtil(object):

    def __init__(self, stopfile, keywords_size, feature_size, one_hot=True):
        self.stopfile = stopfile
        self.keywords_size = None
        self.feature_size = None
        self.stopwords = set()
        self.total_labels = []
        self.label_keywords = {}
        self.feature_weight = {}

#    def load_stopwords()

#    def is_need_delete()

#    def gen_label_keywords()

#    def gen_global_features()

#    def vectorize()
