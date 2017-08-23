#!/usr/bin/env python
# -*- encoding: UTF-8 -*-
# Author: Jerry.Shi
# Date: 2017-08-14 15:12:30

import sys
try:
    from pyfasttext import FastText
except ImportError:
    print("No pyfasttext package found, please check you have install it successfully!")
    sys.exit(-1)

# model train
train_data = '../../data/data.train'
valid_data = '../../data.valid'
model = '../../result/fasttext_model'
classifier = FastText()
classifier.supervised(input=train_data, output=model)

"""
# model evaluation
result = classifier.test(valid_data)
print("P@1: %0.5f" % result.precision)
print("R@1: %0.5f" % result.recall)
print("Number of examples: %d" % result.nexamples)
"""
# prediction
test_corpus = []
with open(valid_data, 'r') as fread:
    for line in fread.readlines():
	line = line.strip()
	if len(line) == 0:
	    continue
	arr = line.split(' , ')
	if len(arr) != 2:
	    continue
	test_corpus.append(arr)

test_data = [x[1] for x in test_corpus]
print("Number of test examples: %d" % len(test_data))
print(test_data[0])

labels = classifier.predict(test_data)
pred_result = zip(test_corpus, labels)
print(labels[:5])
with open('../../result/fasttext_sample.predict', 'w') as f:
    for x in pred_result:
	f.write(x[0][0] + "\t" + x[0][1] + "\t__label__" + x[1][0].encode('utf-8'))
	f.write('\n')

