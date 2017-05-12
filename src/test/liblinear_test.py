#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Author: Jerry.Shi
# Date: 2017-05-12 15:45:30

from liblinearutil import *

# read data in LIBSVM fromat
y, x = svm_read_problem("/data/research/github/ml/liblinear/heart_scale")

# model train
model = train(y[:200], x[:200], '-c 4')

