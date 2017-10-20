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
import datetime

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("input_train_data", "../../../data/gene/data/data.train", 
                    "Input training data")
tf.flags.DEFINE_string("input_dev_data","../../../data/gene/data/dev","Input evaluation data")
tf.flags.DEFINE_string("input_test_data", "../../../data/gene/data/test", "Input text data")
tf.flags.DEFINE_string("out_dir", "../../../result/textcnn", "Training output model dir")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Number of checkpoints to store")

def main(_):
    print("Loading data...")
    start_time = time.time()
    
    # process train/dev/test data
    x_train, y_train, x_dev, y_dev, x_test, y_test, words = data_process(FLAGS.input_train_data,
            FLAGS.input_dev_data, FLAGS.input_test_data)
    
    # config
    model_config = ModelConfig()
    model_config.vocab_size = len(words) + 1
    model_config.filter_sizes = list(map(int, model_config.filter_sizes.split(",")))
    model = TextCNN(model_config)
    end_time = time.time()
    print("Process data and create model time usage: {}s".format(timedelta(seconds=int(round(end_time - start_time)))))
    
    # Constructing Tensorflow Graph
    with tf.Session() as sess:
        model = TextCNN(model_config)
        # define training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(model_config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # keep track of gradient values and sparsity
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "runs", timestamp))
        print("Writing to {}".format(out_dir))

        # summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuarcy)

        # train summary
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(FLAGS.out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # dev summary
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(FLAGS.out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # checkpoint
        ckpt_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "ckpt"))
        ckpt_prefix = os.path.join(ckpt_dir, "model")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)


        # initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.keep_prob: model_config.dropout_keep_prob}
            _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step,train_summary_op, model.loss, model.accuarcy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step: {}, loss:{:g}, accuracy:{:}".format(time_str, step, loss, accuracy))

        def dev_step(x_batch, y_batch, writer=None, test=False):
            feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.keep_prob: 1.0}
            if not test:
                step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, model.loss, model.accuarcy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step: {}, loss:{:g}, accuracy:{:}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
            else:
                loss, accuracy = sess.run(
                        [model.loss, model.accuarcy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step: {}, loss:{:g}, accuracy:{:}".format(time_str, step, loss, accuracy))

        # generate batch data
        batch_train= batch_iter(list(zip(x_train, y_train)), model_config.batch_size,
                model_config.epochs)
        for batch in batch_train:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % model_config.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
                print("\nTest:")
                dev_step(x_test, y_test,test=True)
            
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, ckpt_prefix, global_step=current_step)
                print("Saved model ckeckpoint to {}\n".format(path))


if __name__ == "__main__":
    tf.app.run()
