#!/usr/bin/env python
# -*- encoding: UTF-8 -*-
# Author: Jerry.Shi
# Date: 2017-08-14 15:12:30

import sys,getopt
try:
    from pyfasttext import FastText
except ImportError:
    print("No pyfasttext package found, please check you have install it successfully!")
    sys.exit()

# parsed parameters from cmd
argv = sys.argv[1:]
data_pth = ""
outdir = ""

try:
    opts,args = getopt.getopt(argv, "hi:o:", ["corpus=", "outdir="])
except getopt.GetoptError:
    print("You should input as: py-exe -i corpus -o outdir")
    sys.exit()

for opt, arg in opts:
    if opt == '-h':
        print("You should input as: py-exe -i corpus -o outdir")
    elif opt in ('-i', '--corpus'):
        data_pth = arg
    elif opt in ('-o', '--outdir'):
        outdir = arg
    else:
        print("You should input as: py-exe -i corpus -o outdir")
if len(data_pth) == 0 and len(outdir) == 0:
    print("You should input as: py-exe -i corpus -o outdir")
    sys.exit()
   
print("Input corups:{}, Outdir:{}".format(data_pth, outdir))
        
# get data
train_data = data_pth + '/data.train'
valid_data = data_pth + '/data.valid'
model = outdir + '/fasttext_model'
predict_result = outdir + '/fasttext_predict.txt'

# model train
classifier = FastText()
classifier.supervised(input=train_data, output=model)

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
# write predict result to file
with open(predict_result, 'w') as f:
    for x in pred_result:
	f.write(x[0][0] + "\t" + x[0][1] + "\t__label__" + x[1][0].encode('utf-8'))
	f.write('\n')
print("Model train completed!")
