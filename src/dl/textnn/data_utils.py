#!/usr/bin/env python
# -*- encoding: UTF-8 -*-
# Author: Jerry.Shi
# Date: 2017-10-19 08:39:30

import codecs
import collections import Counter
import tensorflow .contrib.keras as kr
import numpy as np

def read_file(filename):
    """Load training data from file and generate label,sentences list"""
    sentences = []
    labels = []
    with codecs.open(filename) as f:
        for line in f.readlines():
            try:
                arr = line.rstrip().lstrip().split('\t')
                # TODO, use jieba to split strings, here our sample was tokenized
                labels.append(arr[0])
                sentences.append(arr[1].split(' '))
            except:
                pass
    return sentences, labels

def create_cate_dict(labels):
    """Create category dict category to category id.
    Args:
        labels: labels extract from training data
    Return:
        cate_to_id: category to id dictionary
    """
    labels = list(set(labels))
    cate_to_id = dict(labels, range(len(labels)))
    print("create catgegory dict size: {}".format(len(cate_to_id)))
        

def create_vocab(sentences, min_word_count=2):
    """Create vocabulary of word to word_id.
    The vocabulary is saved to disk in a text file of word counts.
    Args:
        sentences: A list of list sentence strings
    Returns:
        A Vocabulary object.
    """
    print("Creating vocabulary.")
    counter = Counter()
    for c in sentences:
        counter.update(c)
    print("Total words: {}".format(len(counter)))

    # filter uncommon words and  sort by descending count
    word_counts = [x for x in counter.items() if x[1] > min_word_count] 
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary: {}".format(len(word_counts)))

    # write out the word counts file
    with codecs.open("../../../result/textcnn/word_counts.txt", 'w') as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("Wrote vocabulary file!")

    # create vocabulary dictionary
    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(i, w) for w, i in enumerate(word_counts)])
    vocab = Vocabulary(vocab_dict, unk_id)

    return vocab

class Vocabulary(object):
    """Vocabulary wrapper"""
    def __init__(self, vocab, unk_id):
        """Initializes vocabulary wrapper
        Args:
            vocab: a dictionary of word to word_id
            unk_id: id of the unknown word"""
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        """Return the word id"""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id

    def id_to_word(self, idx):
        """Return the word string of the id"""
        reverse_vocab = {i: w for w,i in self._vocab.item()}
        if idx in reverse_vocab:
            return reverse_vocab[idx]
        else:
            return ""

def file_to_ids(filename, vocab, max_length=600):
    """Convert text to ids.
    Args:
        filename: data file
        vocab: word_to_id dictionary
        max_length: maximum length of sentences
    Return:
        padding_x: padding ids list
        padding_y: padding label list
    """
    sentences, labels = read_file(filename)
    word_to_id = create_vocab(sentences)
    cate_to_id = create_cate_dict(labels)
    
    # convert all words to their ids
    data_id = []
    label_id = []
    for i in xrange(len(sentences)):
        data_id.apend([ word_to_id[w] for w in sentences[i]])
        label_id.append(cate_to_id[labels[i]])

    # pad sentences, convert label to one-hot
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id)

    return x_pad, y_pad

# TODO: To optimize
def batch_iter(data, batch_size=64, num_epochs=5):
    """Generate batch data."""
    data = np.array(data) 
    data_size = len(data)
    num_batchs_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in xrange(num_epochs):
        indices = np.random.permutation(np.arrange(data_size))
        shuffled_data = data[indices]

        for batch_num in xrange(num_batchs_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index: end_index]

