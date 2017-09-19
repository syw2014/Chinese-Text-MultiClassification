#!/usr/bin/env python
# -*- encoding: UTF-8 -*-
# Author: Jerry.Shi
# Date: 2017-09-15 15:07:30


# Text Classification
import sys, getopt
import codecs
import numpy as np

# import pyfasttext package
try:
    from pyfasttext import FastText
except ImportError:
    print("No pyfasttext package found, pleach check you have installed it successfully!")
    sys.exit()


def train(train_data, model_name):
    classifier = FastText()
    classifier.supervised(input=train_data, output=model_name, lr=0.7)#, epoch=20, lr=0.7)
    return classifier


def batch_predict(test_data, classifier):
    labels = classifier.predict(test_data)
    return labels

def evaluate(raw_labels, labels, predict):
    assert (len(labels) == len(predict), "Require the same size of test sample")
    # extract labels
    preds = []
    for i, x in enumerate(predict):
        if len(x) == 0:
            print x,i
            continue
        preds.append(int((x[0]).encode('utf-8')))
    predict = preds
    # calculate sample size of each label
    print predict[:5]
   
    # create 
    # create id->label, label->id
    id_to_label = dict( zip(range(len(raw_labels) ), raw_labels ))
    label_to_id = dict( zip(raw_labels, range(len(raw_labels)) ) )

    # sample-> idx, idx->sample
    num_samples = len(labels)
    sample_to_id = dict( zip(labels, range(num_samples)) )
    id_to_sample = dict( zip( range(num_samples), labels) )


    sample_per_label = {}
    for x in id_to_sample.items():
        if sample_per_label.has_key(x[0]):
            sample_per_label[x[0]].append(x[1])
        else:
            sample_per_label[x[0]] = [ x[1] ]

    # create confusion matrix
    cofus_mat = np.zeros((len(raw_labels), len(raw_labels)), dtype=np.int32)
    num_pos = 0
    for i, x in enumerate(predict):
        true_label = id_to_sample[i]            # get the truth label
        if predict[i] not in label_to_id or true_label not in label_to_id:
            continue
        true_label_id = label_to_id[true_label] # get true label id
        label_id = label_to_id[ predict[i] ]    # get predict label idx
        # true sample
        if true_label == predict[i] :
            cofus_mat[ label_id][label_id] += 1
            num_pos += 1
        else: # false predict
            cofus_mat[ true_label_id][label_id] += 1

    # calculate precision, recall, fb1
    for i in xrange(len(cofus_mat[0])):
        # numbers of samples in each label, number of predict samples in each label
        row_sum, col_sum = sum(cofus_mat[i]), sum(cofus_mat[c][i] for c in range(len(cofus_mat)))
        if cofus_mat[i][i] == 0:
            print("Label: {} Precision: {} Recall: {} FB1: {} test sample number: {}".format(id_to_label[i], 0., 0., 0., row_sum))
        else:
            p = cofus_mat[i][i] / float(col_sum)
            r = cofus_mat[i][i] / float(row_sum)
            print("Label: {} Precision: {:.3f} Recall: {:.3f} FB1: {:.3f} test samples number: {}".format(id_to_label[i], (cofus_mat[i][i])/float(col_sum), \
                cofus_mat[i][i] / float(row_sum), 2*p*r / float(p+r), row_sum ))
    print("Total accuracy: {:.3f}".format( num_pos/ float(len(predict))))

def main():
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, 'hi:o:', ['corpus=', 'outdir='])
    except getopt.GetoptError:
        print('Usage: py-exe -i corpus -o outdir')
        sys.exit()
    data_dir, outdir = "", ""
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: py-exe -i corpus -o outdir')
            sys.exit()
        elif opt in ['-i', '--corpus']:
            data_dir = arg
        elif opt in ['-o', '--outdir']:
            outdir = arg
        else:
            print('Usage: py-exe -i corpus -o outdir')
            sys.exit()

    if len(data_dir) == 0 and len(outdir) == 0:
        print('Usage: py-exe -i corpus -o outdir')
        sys.exit()
    print("Input data dir: {}, output dir: {}".format(data_dir, outdir))

    # get train data
    train_data = data_dir + '/data.train'
    val_data = data_dir + '/data.val'
    # model train
    classifier = train(train_data, outdir + '/model')
    test_corpus = list()
    with codecs.open(val_data) as f:
        for line in f.readlines():
            line = line.strip()
            arr = line.split(' , ')
            if len(arr) != 2:
                continue
            test_corpus.append(arr)
    print("Load test data completed size: {}".format(len(test_corpus)))

    # predict
    test_data = [x[1] for x in test_corpus]
    test_labels = [int(x[0].lstrip("__label__")) for x in test_corpus]
    raw_label = list()
    with codecs.open(train_data) as f:
        for line in f.readlines():
            arr = line.split(' , ')
            if len(arr) != 2:
                continue
            raw_label.append(arr[0])
    raw_label = [ int(x.lstrip("__label__")) for x in raw_label ]
    raw_label = list(set(raw_label))
    labels = batch_predict(test_data, classifier)
    print("predict size: {}".format(len(labels)))
    # evaluate
    evaluate(raw_label, test_labels, labels)
    # write predict result
    pred_result = zip(test_corpus, labels)
    with codecs.open('predict.txt', 'w') as f:
        for x in pred_result:
            if len(x[1]) == 0:
                continue
            f.write(x[0][0] +  "\t" + x[0][1] +  "\t__label__" + (x[1][0]).encode('utf-8'))
            f.write('\n')
            
            #print(x[0][0] +  "\t" + x[0][1] +  "\t__label__" + x[1][0].encode('utf-8'))
    print("Model process completed!")


if __name__ == "__main__":
    main()
