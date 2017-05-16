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


def is_need_delete(term):
    """Check term is satisfy delete conditions.
    term: input keywords
    """
    # remove white space in the front and back
    term = term.rstrip().lstrip()

    # rule1, length <= 1 or length > 30
    lenth = len(term)
    if lenth <= 1 or lenth > 30:
        return True
    # rule2, is digital
    if term.isdigit():
        return True

    # rule3, startswith or endswith '_'
    if term.startswith('_') or term.endswith('_'):
        return True

    # rule4, startswith '&'
    if term.startswith('&'):
        return True
    return False

class DataUtil(object):
    def __init__(self, keywords_num, feature_num, one_hot=True):
        """Construct a DataSet.
        keywords_num: the number of keywords in each label
        feature_num: the number of features for the constructed dataset
        one_hot: used only for how to convert text to a numerical vector
        """
        self.keywords_num = keywords_num
        self.feature_num = feature_num
        self.stopwords = set()  # stop words set
        self._labels = None  # all the labels appears in corpus
        self._labels_keywords = None  # keywords in each label, key is label, value is keywords list
        self._feature_weight = None  # global feature weight for the whole corpus

    def labels(self):
        """Get all labels appears in dataset."""
        return self._labels

    def label_keywords(self):
        """Get label keywords list."""
        return self._labels_keywords

    def features(self):
        """Get feature words and it's weight."""
        return self._feature_weight

    def load_stopwords(self, filename):
        """Load stop words into 'self.stopwords' set
        filename: stopwords filename
        """
        # check file is valid
        if len(filename) == 0:
            print("Error: require non-empty stopwords, but found %s" % filename)
            sys.exit()
        self.stopwords = set()  # self.stopwords = ()
        print("Start load stop words...")

        with open(filename, 'r') as ifs:
            for line in ifs.readlines():
                line = line.rstrip().lstrip()
                self.stopwords.add(line)
        print("Stopwords loaded, total number: %d" % len(self.stopwords))

    def gen_corpus_features(self, corpus):

        """Given a corpus with label samples, select features for input corpus.
        corpus: input corpus, structure should be : 'label_id \t txt', tokens delimiter must be 'space'
        """
        if len(corpus) == 0:
            print("Error: require non-corpus, but found %s" % corpus)
            sys.exit(-1)
        # step1, load label data
        label_samples = {}
        sample_num = 0
        self._labels = set()
        start_t = time.time()
        with open(corpus, 'r') as ifs:
            for line in ifs.readlines():
                line = line.rstrip().lstrip()
                cxt = line.split("\t")
                if len(cxt) != 2:
                    continue
                label_id = int(cxt[0].strip())
                sample = cxt[1].lstrip()

                sample_num += 1

                # store data
                if label_samples.has_key(label_id):
                    label_samples[label_id].append(sample)
                else:
                    label_samples[label_id] = [sample]
        end_t = time.time()
        print("Load samples completed, label number: %d, duration: %.3fs" % (len(label_samples), (end_t - start_t)))

        # get all keys in corpus
        self._labels = label_samples.keys()
        # --------------TODO: data unbalance process ----------------
        # choose weight process to deal unbalance
        # weight = max_label_sample_num / ( label_sample_num + 1.0)

        label_sample_num = {}  # sample number of each label
        for k, v in label_samples.items():
            label_sample_num[k] = len(v)
#            print("Debug -> length: %d" % len(v))
        # get maximum samples number in all labels
        max_label_sample_num = np.array(label_sample_num.values()).max()
        # compute sample weight for each label
        label_sample_w = {}
        for k, v in label_sample_num.items():
            label_sample_w[k] = max_label_sample_num / (v + 1.0)
        # ------------------------------------------------------------

        # extract keywords for each labels
        start_t = time.time()
        self._labels_keywords = {}  # key: label_id, val: keyword list
        for label_id, samples in label_samples.items():
            term_tf = {}  # key: term, val: term frequency
            for txt in samples:
                txt = txt.rstrip().lstrip()
                if len(txt) == 0:
                    continue
                tokens = txt.upper().split(" ")
                for tok in tokens:
                    # remove stop words
                    if tok in self.stopwords:
                        continue
                    # remove words statisfy conditions
                    if is_need_delete(tok):
                        continue
                    if term_tf.has_key(tok):
                        term_tf[tok] += 1  # frequency + 1
                    else:
                        term_tf[tok] = 1
            # sort by term frequency
            term_tf = sorted(term_tf.iteritems(), key=lambda d: d[1], reverse=True)
            # select topk words as label keywords
            # TODO: store keywords TF for other tasks
            keywords = [x[0] for x in term_tf][: self.keywords_num]
            # reserve keywords list
            self._labels_keywords[label_id] = keywords
        end_t = time.time()
        print("Label keywords generation completed, duration: %.3fs" % (end_t - start_t))

        # generate global features for input corpus
        # In this case, we use IDF as the selection rules to select feature
        self._feature_weight = {}
        docs = self._labels_keywords.values()  # get all keywords
        label_num = len(self._labels_keywords)  # total doc number
        feature_w = {}
        start_t = time.time()
        for _, tokens in self._labels_keywords.items():
            for word in tokens:
                df = 0
                # compute doc freq in all docs
                for i in xrange(len(docs)):
                    if word in docs[i]:
                        df += 1

            if feature_w.has_key(word):
                continue
            else:
                feature_w[word] = math.log(label_num / (df + 1.0))
                # TODO, use TFIDF as feature weight
        # select topk features
        feature_w = sorted(feature_w.iteritems(), key=lambda d: d[1], reverse=True)
        feats = [x[0] for x in feature_w][: self.feature_num]
        weis = [x[1] for x in feature_w][: self.feature_num]
        self._feature_weight = zip(feats, weis)
        end_t = time.time()
        print("Corpus features generation completed, duration: %.3fs" % (end_t - start_t))

    def vectorize(self, sample, sparse=True):

        """Text string one-hot encoding, convert a text into a numeric vector for ML/DL model training.
        sample: input corpus, structure should be, 'label_id \t txt', tokens delimiter must be 'space'
        return: int(label),vector
        """
        tmp = sample.rstrip().lstrip().upper().split("\t")
        if len(tmp) != 2:
            print("Error: require two fields but found %s" % sample)
            sys.exit(-1)
        label_id = int(tmp[0].strip())
        tokens = tmp[1].rstrip().split(" ")

        # compute term frequency
        term_tf = {}
        for tok in tokens:
            if term_tf.has_key(tok):
                term_tf[tok] += 1
            else:
                term_tf[tok] = 1

        # extract features and weights
        features = [x[0] for x in self._feature_weight]
        weights = [x[1] for x in self._feature_weight]

        # vectorize
        feature_dim = len(self._feature_weight)
        dense_vector = np.zeros(feature_dim)
        sparse_vector = {}
        for k, tf in term_tf.items():
            if k in features:
                idx = features.index(k)
                wei = weights[idx]
                if sparse:
                    sparse_vector[idx+1] = tf * wei  # index + 1 to statisfy libsvm data format
                else:
                    dense_vector[idx] = tf * wei
            else:
                pass

        # if encoding vector is empty, return directly
        # added the max index into vector
        if sparse:
            """
            if len(sparse_vector) == 0:
                return label_id, sparse_vector
            midx = len(features) - 1
            if not sparse_vector.has_key(midx):
                sparse_vector[midx] = 0.0
            """
            return label_id, sparse_vector
        else:
            return label_id, dense_vector  

    def batch_data(self, data, sparse=True):
        """Batch encoding.
        data: input file name
        sparse: specify vector is dense or sparse, default is sparse.
        return: labels array, size is data size, arrary(dense, element: weight) or dict(sparse, key:index, val: weight)
        """
        if len(data) == 0:
            print("Error: require non-empty data, but found %s" % data)
            sys.exit(-1)
        labels = []
        X_vec = []
        with open(data, 'r') as ifs:
            for line in (ifs.readlines()):
                #time.sleep(0.01)
                lidx, vec = self.vectorize(line, sparse)
                if len(vec) == 0:
                    continue
                labels.append(lidx)
                X_vec.append(vec)
        return labels, X_vec
