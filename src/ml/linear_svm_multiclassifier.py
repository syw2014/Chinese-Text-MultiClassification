
# coding: utf-8

# In[13]:

#!/usr/bin/env python

# Author : jerry.Shi
# Description:
#      A multi-class classification  based on linear svm, it can be uesed in the produce environment.

import sys
import numpy as np
import math
from __future__ import division

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
    

# step 1, generate feature words
def gen_feature_words(data, topk=10000):
    if len(data) == 0:
        print("Require a non-empty file, but found %s", data)
        sys.exit(-1)
    
    # aggerate trainning samples by key, key : label_id, value: training samples
    label_samples = {}
    total_sample_size = 0
    with open(data, 'r') as ifs:
        for line in ifs.readlines():
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
    k_words = 100
    vocab = {} # key: label id, val : feature words
    for label, samples in label_samples.items():
        token_tf = {}
        for txt in samples:
            txt = txt.strip()
            if len(txt) == 0:
                continue
            cxt = txt.split(" ")
            # add token to token_tf
            for tok in cxt:
                if len(tok) < 1:
                    continue
                if token_tf.has_key(tok):
                    token_tf[tok] += 1  # tf + 1
                else:
                    token_tf[tok] = 1
        # sort by tf 
        token_tf = dict(sorted(token_tf.iteritems(), key = lambda d:d[1], reverse=True))
        # select k words as feature 
        feature_words = token_tf.keys()[: k_words]
        
        # insert label feature words into global vocabulary
        vocab[label] = feature_words
    
    # select global feature based on inverse label frequency
    global_features = {}  # key: term, value: idf
    words = vocab.values()
    label_size = len(vocab) # label number
    for label, tokens in vocab.items():

        for tok in tokens:
            df = 0
            i = 0
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
    global_features = dict(sorted(global_features.iteritems(), key= lambda d:d[1], reverse=True))
    features = global_features.keys()[:featue_size]
    weights = global_features.values()[:featue_size]
    print("global feature generation completed, total size: %d") % (len(features))
    # final global features
    tmp = dict(zip(features, weights))
    with open('test.txt', 'w') as ofs:
        for k, v in tmp.items():
            ofs.write(k + "\t" + str(v))
            ofs.write('\n')
            
    return dict(zip(features, weights))
    
    
            
    
# step 2, vectorize training sample
# step 3, training classifier
# step 4, predict with probability

# main 
if __name__ == "__main__":
    stopwords_file = "stopwords.txt"
    stop_words = load_stopwords(stopwords_file)
    
    train_sample = "train_test.txt"
    gen_feature_words(train_sample, 20000)


# In[ ]:



