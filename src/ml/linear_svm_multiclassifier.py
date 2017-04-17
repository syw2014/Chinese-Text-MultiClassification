
# coding: utf-8

# In[2]:

#!/usr/bin/env python

# Author : jerry.Shi
# Description:
#      A multi-class classification  based on linear svm, it can be uesed in the produce environment.

import numpy as np

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
    label_sample_num = []
    for x in label_samples.values():
        label_samples.append(x)
    max_label_num = np.array(label_sample_num).max()
    
    
            
    
# step 2, vectorize training sample
# step 3, training classifier
# step 4, predict with probability

# main 
if __name__ == "__main__":
    stopwords_file = "stopwords.txt"
    stop_words = load_stopwords(stopwords_file)


# In[ ]:



