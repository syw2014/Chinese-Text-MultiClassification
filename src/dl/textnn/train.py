#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf

from cnn_model import *
from config import *
from data_utils import *

import time
import os
from datetime import timedelta

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("input_train_data", "../../../data/gene/data/data.train", 
                    "Input training data")
tf.flags.DEFINE_string("input_dev_data","../../../data/gene/data/data.dev","Input evaluation data")
tf.flags.DEFINE_string("input_test_data", "../../../data/gene/data/data.dev", "Input text data")
tf.flags.DEFINE_string("ckpt", "../../../result/textcnn/ckpt", "Training output model dir")


def main(_):
    print("Loading data...")
    start_time = time.time()
    
    if not os.path.exists(FLAGS.ckpt):
        os.makedirs(FLAGS.ckpt)
    # process train/dev/test data
    x_train, y_train, x_dev, y_dev, x_test, y_test, words = data_process(FLAGS.input_train_data,
            FLAGS.input_dev_data, FLAGS.input_test_data)
    
    # config
    model_config = ModelConfig()
    model_config.vocab_size = len(words) + 1
    model = TextCNN(model_config)
    
    # tensorboard_dir
    tensorboard_dir = '../../../result/tensorboard/textcnn'
    
    end_time = time.time()
    print("Process data and create model time usage: {}s".format(timedelta(seconds=int(round(end_time - start_time)))))
    
    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.ckpt):
        os.makedirs(FLAGS.ckpt)
    save_path = os.path.join(FLAGS.ckpt, "best_eval")

    # Constructing Tensorflow Graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # config tensorboard
        tf.summary.scalar("loss", model.loss)
        tf.summary.scalar("accuracy", model.accuray)

        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(sess.graph)

        # generate batch data
        batch_train= batch_iter(list(zip(x_train, y_train)), model_config.batch_size,
                model_config.epochs)


        def feed_data(batch):
            """"""
            x_batch, y_batch = zip(*batch)
            feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch
                    }
            return feed_dict, len(x_batch)

        def evaluate(x, y):
            batch_eval = batch_iter(list(zip(x,y)), 128, 1)

            total_loss = 0.0
            total_acc = 0.0
            cnt = 0.0
            for batch in batch_eval:
                feed_dict, cur_batch_len = feed_data(batch)
                feed_dict[model.keep_prob] = 1.0
                loss , acc = sess.run([model.loss, model.accuray], feed_dict=feed_dict)
                cnt += cur_batch_len

            return total_loss / cnt, total_acc / cnt


        # start training
        print("Model training")
        print_per_batch = model_config.print_per_batch
        for i, batch in enumerate(batch_train):
            feed_dict , _ = feed_data(batch)
            feed_dict[model.keep_prob] = model_config.dropout_keep_prob
    
            if i % 5 == 0:
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, i)

            if i % print_per_batch == print_per_batch - 1:
                loss_train, acc_train = sess.run([model.loss, model.accuray], 
                        feed_dict=feed_dict)
                loss, acc = evaluate(x_dev, y_dev)

                print("Iter: {0:>6}, Train loss: {1:>6.2}, Train Acc: {2:>7.2%} Val loss: {3:6.2}, Val Acc:{4:7.2%}"
                        .format(i+1, loss_train, acc_train, loss, acc))

                loss_test, acc_test = evaluate(x_test, y_test)
                print("Test loss: {0:6.2}, Test Accuracy: {1:>7.2%}".format(loss_test, acc_test))
            sess.run(model.optimize, feed_dict=feed_dict)

        # model test
        print("Model evaluation on test set...")
        loss_test, acc_test = evalute(x_test, y_test)
        print("Test loss: {0:6.2}, Test Accuracy: {1:>7.2%}".format(loss_test, acc_test))


if __name__ == "__main__":
    tf.app.run()
