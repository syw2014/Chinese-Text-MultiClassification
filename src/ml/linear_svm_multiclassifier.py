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
from liblinearutil import *

# add local path to sys path
sys.path.append("../util/")
from data_util import *

    

# main 
if __name__ == "__main__":
    stopwords_file = "/data/research/data/dict/stopwords.txt"
    dataUtil = DataUtil(200, 10000)
    dataUtil.load_stopwords(stopwords_file)
    
    train_sample = "/data/research/data/cn_classify/text_2w.train"
    dataUtil.gen_corpus_features(train_sample)
    labels = dataUtil.labels()
    label_keywords = dataUtil.label_keywords()
    features = dataUtil.features()
    dirs = "../../result/data"
    with open(dirs + '/label.txt', 'w') as ofs1, open(dirs+'/label_feature.txt', 'w') as ofs2, open(dirs+'/features.txt', 'w') as ofs3:
        for x in labels:
            ofs1.write(str(x))
            ofs1.write('\n')
        for k, v in label_keywords.items():
            ofs2.write(str(k)+ "\t" + " ".join(v))
            ofs2.write('\n')
        for x in features:
            ofs3.write(x[0] + '\t' + str(x[1]))
            ofs3.write('\n')
    """ 
    y_, X = dataUtil.batch_data(train_sample)
    
    print("sample num: %d, label num: %d" % (len(X), len(y_)))
    # linear model train
    prob = problem(y_, X)
#    print X
    param = parameter('-s 0 -c 4 -B 1')
    m = train(prob, param)
    save_model("../../result/model/linear_classify.model", m)
    
    """
    # predict
    pre_file = "/data/research/data/cn_classify/predict.txt"
    y_, X = dataUtil.batch_data(pre_file)
    m = load_model("../../result/model/linear_classify.model")
    p_label, p_acc, p_val = predict(y_, X, m, '-b 1')
    ACC, MSE, SCC = evaluations(y_, p_label)
    
