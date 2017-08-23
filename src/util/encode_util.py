#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : encode_util.py
# PythonVersion: python3.5
# Date    : 2017/5/31 10:31
# Software: PyCharm Community Edition
from tqdm import *

try:
    import re2 as re
except ImportError:
    import re

import sys, getopt

# Data normalization, to remove special characters, duplicated entries, collect all attributes for the same keyword, and
# Normalize the specific attributes to the standard words,and write the AVPs in one line split by '&&'

# usefull tools
# unicode is chinese
def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fff':
        return True
    else:
        return False

# unicode is numerical
def is_number(uchar):
    if u'\u0030' <= uchar <= u'\u0039':
        return True
    else:
        return False

# unicode is alphabet
def is_alphabet(uchar):
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False

# unicode is not chinese , not numerical, not alphabet
def is_other(uchar):
    if not (is_chinese(uchar) or is_alphabet(uchar) or is_number(uchar)):
        return True
    else:
        False

# common interface
def main(argv):
    info_file = ''
    outfile = ''
    attr_norm_file = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:")
    except getopt.GetoptError:
        print "python-exec -i <inputfile> -o <outfile>"

    # parse
    for opt, arg in opts:
        if opt == '-h':
            print "python-exec -i <inputfile> -o <outfile>"
            sys.exit()
        elif opt == "-i":
            info_file = arg
        elif opt == "-o":
            outfile = arg
        else:
            print "python-exec -i <inputfile> -o <outfile>"
            sys.exit()
    print("Parameters parsed input: %s, output: %s" % (info_file, outfile))

