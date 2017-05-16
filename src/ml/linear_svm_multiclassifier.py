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

# add local path to sys path
sys.path.append("../util/")
from data_util import *

# step 1, 

    

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

    with open('../../result/label.txt', 'w') as ofs1, open('../../result/label_feature.txt', 'w') as ofs2, open('../../result/features.txt', 'w') as ofs3:
        for x in labels:
            ofs1.write(str(x))
            ofs1.write('\n')
        for k, v in label_keywords.items():
            ofs2.write(str(k)+ "\t" + " ".join(v))
            ofs2.write('\n')
        for x in features:
            ofs3.write(x[0] + '\t' + str(x[1]))
            ofs3.write('\n')

    y_, X = dataUtil.batch_data(train_sample)

