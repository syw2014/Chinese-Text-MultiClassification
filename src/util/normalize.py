#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : Normalize.py
# PythonVersion: python3.5
# Date    : 2017/5/31 10:31
# Software: PyCharm Community Edition
# from __future__ import unicode_literals
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


def data_clean(infile):
    '''
    load data and parse info dict and title synonyms dict
    :param infile: data structure in file should be 'keyword \t attribute \t value'
    info: avp dict(key, dict(attr, val)), key: keyword of entry, attr: attribute, val: value
    title_syns: the synonym dict(title, list[word])
    :return: info, title_syns
    '''
    info = {}
    title_syns = {}

    # There are two situations when cleaning attribute values, english and chinese because they have different process
    # use english punctuation to clean attribute values
    eng_chars = ('~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '=',
                     ';', ':', '"', '\'', '\\', '|', '{', '[', '}', ']', '<', ',', '>', '.', '?', '/',
                     )
    # we use chinese punctuation to clean attribute values
    cn_chars = [ '·', '，',  '。', '》', '？', '、', '“', '”', '；', '：',  '】', '——',
                    '￥', '……',  '）'
                    ]
    special_chars = ''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠ \
                    々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻ \
                    ︽︿﹁﹃﹙﹛﹝（｛“‘-—_… ~`@#$%^&*()_-+=?/][{};:：,.<>?.'''

    special_chars2 = ''':!),.:;?]}`@#$%^&*()_-+=?/][{};:：,.<>?.'''

    digit_pat = r"^\d*"

    with open(infile, 'r') as ifs:
        for line in tqdm(ifs.readlines()):
            line = line.lstrip().rstrip()
            cxt = line.split("\t")
            if len(cxt) < 3:
                print("Warning: %s" % line)
                continue
            keyword = cxt[0].strip().lstrip()
            attr = cxt[1].lstrip().rstrip()
            val = cxt[2].lstrip().rstrip()

            # ------ attribute clean -------
            # rule1, remove all english punctuation appears in attributes
            # print attr
            # if re.findall('[\x80-\xff].', attr):
            #     attr = attr.translate(None, special_chars2)
            # another way to remove special chars

            attr =attr.decode('utf-8')
            for e in attr:
                if is_other(e):
                    attr = attr.replace(e, u'')
            # print attr

            # rule2, check digit
            result = re.findall(digit_pat, attr)
            if len(result) == 1 and result[0] != '':
                substr = attr[len(result[0]):]
                # new attribute
                attr = substr + result[0]

            # ------ attribute value clean -------
            # rule1, remove special chars in the start and end
            while True:
                # start with english punctuation
                if val.startswith(eng_chars):
                    val = val[1:]
                    continue
                elif val.endswith(eng_chars):  # end with english punctuation
                    val = val[:len(val) - 1]
                else:
                    break
            # rule2, remove all chinese punctuation appears in attributes values
            for char in cn_chars:
                cpos = val.find(char)
                if cpos != -1:
                    # val = val.replace(char, '')
                    if cpos == 0:
                        val = val[3: ]   # as the chinese encoding occupy three bytes ,so start with the fourth position
                    elif cpos == -1:
                        val = val[: len(val) - 3]

            # collect synonym words
            if attr == "title":
                if title_syns.has_key(val):
                    title_syns[val].add(keyword)
                else:
                    title_syns[val] = set([keyword])
            else:
                if info.has_key(keyword):
                    info[keyword].add((attr, val))
                else:
                    info[keyword] = set([(attr, val)])
    print("data info loaded size: %d, sysnonym collected size: %d" % (len(info), len(title_syns)))
    return info, title_syns


def normalize(info, sysnonym, attr_norm_file,outfile):
    '''
    Normalize attribute words.
    :param info: attribute value pairs dict, key: keyword, val: (attr, val)
    :param sysnonym: synonym of title keywords
    :param attr_norm_file attribute normalization word
    :param outfile: the final ourput file
    :return: None
    '''
    # load attr norm words
    attr_norm_words = {}
    with open(attr_norm_file, 'r') as ifs:
        for line in ifs.readlines():
            line = line.strip()
            cxt = line.split("\t")
            if len(cxt) == 1:
                continue
            if attr_norm_words.has_key(cxt[0].strip()):
                for ele in cxt[1:]:
                    attr_norm_words[cxt[0].strip()].add(ele)
            else:
                attr_norm_words[cxt[0].strip()] = set(cxt[1:])
    print("attribute norm words loaded size: %d" % len(attr_norm_words))

    # normalize
    print("Start normalizating...")
    with open(outfile, 'w') as ofs:
        for keyword, synons in sysnonym.items():
            if info.has_key(keyword):
                # the same entry
                sysn_str = keyword
                for syn in synons:
                    sysn_str += "\t" + syn

                # attribute convertion
                avp_new = []
                avp_str = ""
                raw_avps = info[keyword]
                for ele in raw_avps:
                    new_attr = ""
                    for attr_key, attr_val in attr_norm_words.items():
                        if ele[0] in attr_val:
                            new_attr = attr_key
                            break
                    if new_attr != "":
                        avp_str += "\t"+ new_attr + ':' + ele[1] + ':' + ele[0].encode('utf-8')   # new_attr : value : synonym
                    else:
                        avp_str += "\t" + ele[0].encode('utf-8') + ':' + ele[1]
                ofs.write(sysn_str + "&&" + avp_str)
                ofs.write("\n")
    print("Data normalization competed!")

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



if __name__ == "__main__":
    info_file = "D:\\work\\Project\\Tasks\\6 Knowlege graph\\data\\baike\\result\\result.txt"
    outfile = "D:\\work\\Project\\Tasks\\6 Knowlege graph\\data\\baike\\result\\norm_game_info2.txt"
    attr_norm_file = "D:\\github\\kg-construction\\data\\attr_norm_words.txt"


    info, title_syns = data_clean(info_file)
    normalize(info, title_syns, attr_norm_file, outfile)
