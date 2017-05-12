#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Author : jerry.Shi
# Description:
#      A multi-class classification  based on linear svm, it can be uesed in the produce environment.

from __future__ import division
import sys
import numpy as np
import math
import scipy
from tqdm import *

# step 0, load filter words for feature generation 
def load_stopwords(filename):
    if len(filename) == 0:
        print("Require a non-empty file, but found %s", filename)
        sys.exit(-1)
    # stop words set for duplicated words
    stop_words = set()
    
    with open(filename, 'r') as ifs:
        for line in ifs.readlines():
            line = line.strip()
            stop_words.add(line)
        print("stop words loaded , total number %d", len(stop_words))
    return stop_words
    
def is_need_del(term):
    """To check the term is needed remove"""
    term = term.strip()
    # rule1, length <1 or length > 30
    if len(term) <= 1 or len(term) >= 30:
        return True
    # rule2, startswith or endswith '_'
    if term.startswith('_') or term.endswith('_'):
        return True
    # rule3, startswith '&'
    if term.startswith('&'):
        return True
    # rule4, isdigit
    if term.isdigit():
        return True
    
    return False
    
    

# step 1, generate feature words
def gen_feature_words(data, stopwords, topk=10000):
    if len(data) == 0:
        print("Require a non-empty file, but found %s", data)
        sys.exit(-1)
    
    # load stop words
    stop_words = load_stopwords(stopwords)
    
    # aggerate trainning samples by key, key : label_id, value: training samples
    label_samples = {}
    total_sample_size = 0
    with open(data, 'r') as ifs:
        for line in tqdm(ifs.readlines()):
            line = line.strip()
            cxt = line.split("\t")
            if len(cxt) !=3 :
                continue
            labelID = cxt[0].strip()
            sample = cxt[1].strip()
            
            # sample size add 1
            total_sample_size += 1
            
            # insert data 
            if label_samples.has_key(labelID):
                label_samples[labelID].append(sample)
            else:
                tmp = [sample]
                label_samples[labelID] = tmp
    print("All training samples loaded, total sample number: %d, total label number: %d") %(total_sample_size, len(label_samples))
    
    # TODO: data unbalance process
    label_sample_num = {}
    for k, v in label_samples.items():
        label_sample_num[k] = len(v)
    
    # get maximum samples number of label
    max_label_num = np.array(label_sample_num.values()).max()
    
    # compute label sample weight based on training samples number
    label_sample_weight = label_sample_num
    for k, v in label_sample_num.items():
        label_sample_weight[k] = (max_label_num) / (v + 1.0)

    # select feature words for every label based on term frequency
    k_words = 200
    vocab = {} # key: label id, val : feature words
    for label, samples in label_samples.items():
        token_tf = {}
        for txt in samples:
            txt = txt.strip()
            if len(txt) == 0:
                continue
            # convert all letter to upper
            cxt = txt.upper().split(" ")
            # add token to token_tf
            for tok in cxt:
                if tok in stop_words:
                    continue
                
                if token_tf.has_key(tok):
                    token_tf[tok] += 1  # tf + 1
                else:
                    token_tf[tok] = 1
        # sort by tf 
        token_tf = sorted(token_tf.iteritems(), key = lambda d:d[1], reverse=True)
        # select k words as feature
        feature_words = []
        for i in range(len(token_tf)):
            if i >= k_words:
                break
            feature_words.append(token_tf[i][0])
        
        # insert label feature words list into global vocabulary
        vocab[label] = feature_words
    
    # select global feature based on inverse label frequency
    global_features = {}  # key: term, value: idf
    words = vocab.values()
    label_size = len(vocab) # label number
    for label, tokens in vocab.items():
        for tok in tokens:
            df = 0
            # calculate document frequency
            for i in range(len(words)):
                if tok in words[i]:
                    df += 1
            # add word to global feature dict
            if global_features.has_key(tok):
                continue
            else:
                idf = math.log(label_size / (df + 1.0))
                global_features[tok] = idf
    
    # select topk term as global feature
    featue_size = 20000
    global_features = sorted(global_features.iteritems(), key= lambda d:d[1], reverse=True)
    features = []
    weights = []
    for idx in xrange(len(global_features)):
        features.append(global_features[idx][0])
        weights.append(global_features[idx][1])
    print("global feature generation completed, total size: %d") % (len(features))
    # final global features
    tmp = dict(zip(features, weights))
    with open('test.txt', 'w') as ofs:
        for k, v in tmp.items():
            ofs.write(k + "\t" + str(v))
            ofs.write('\n')
            
    return dict(zip(features, weights))
        
# step 2, vectorize training sample based on bag of words model, return training data vector
def sample_vectorize(vocab, sample_file): 
    # check samples and gennerated vocabulary
    if len(vocab) == 0 or len(sample_file) == 0:
        print("Error: require no-empty vocba and sample file")
    # loop samples and convert text to vector
    with open(sample_file, 'r') as ifs:
        # get the size of training data
        num_train = len(ifs.readlines())
        print("Debug: num_train-> %d, vocab_size->%d" %(num_train,  len(vocab)))
        # construct training data array: N X M, N is the number of training data, M is the dimension of every point
        train_data = np.zeros((num_train, len(vocab)))
        train_label = np.zeros(num_train)
        # loop samples and process
        j = 0  # loop variable for samples
        for line in tqdm(ifs.readlines()):
            # require j < number of training data
            if j >= num_train :
                break
            line = line.strip()
            cxt = line.split('\t')
            if len(cxt) < 3:
                continue
            label = int(cxt[0]) # extract label
            tokens = cxt[1].strip()
            # TODO: remove sample length < 3
            # calculate term frequency
            term_tf = {}
            for tok in tokens:
                if term_tf.has_key(tok.strip()):
                    # frequency + 1
                    term_tf[tok] += 1
                else:
                    term_tf[tok.strip()] = 1
            vec = np.zeros(len(vocab))
            i = 0
            for word, weight in vocab.items():
                # require i < the number of feature
                if i >= len(vocab):
                    break
                if word in term_tf:
                    vec[i] = term_tf[word] * weight
                    i+= 1
                else:
                    vec[i] = 0
                    i += 1
            # store every data point 
            train_data[j, :] = vec
            train_label[j] = label
            j += 1
        
        # final 
    print("Vectorization completed, training data shape: ", train_data.shape)
    print("Vectorization completed, training label shape: ", train_label.shape)
    return train_data, train_label

    
# step 3, training classifier
# step 4, predict with probability

# main 
if __name__ == "__main__":
    stopwords_file = "/data/research/data/dict/stopwords.txt"
    stop_words = load_stopwords(stopwords_file)
    
    train_sample = "/data/research/data/cn_classify/train_sample_2w.txt"
    vocab = gen_feature_words(train_sample,stopwords_file , 10000)
    sample_vectorize(vocab, train_sample)

