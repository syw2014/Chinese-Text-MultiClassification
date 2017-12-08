#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Jerry.Shi
# Date: 2017-08-15 18:02:30


from flask import Flask, render_template, request
from pyfasttext import FastText
import json
import codecs

# load model
def load_model(pth):
    if len(pth) == 0:
        return None
    # create object
    model = FastText()
    model.load_model(pth)
    return model

model = load_model('../../result/model.bin')

# load label id
id_to_label = dict()
#def id_to_label(filename):
with codecs.open('../../data/gene/category.txt') as f:
    id_to_label = dict()
    for line in f.readlines():
        line = line.strip()
        arr = line.split('\t')
        id_to_label[arr[0]] = arr[1].decode('utf-8')
print("total label size: {}".format(len(id_to_label)))

# predcit with prob
def predict1(model, sentence):
    result = model.predict_proba_single(sentence)
    if len(result) == 0:
        json_res = {"sentence":sentence, "label": "null", "prob": "0.0"}
    else:
        json_res = {"sentence":sentence, "label": id_to_label[(result[0][0]).encode('utf-8')], "prob": result[0][1]}
    return json_res

# webapp
apps = Flask(__name__)

@apps.route('/api', methods=['GET'])
def index():
    query = request.args.get('query')
    json_res = predict1(model, query)
    print("INFO - %s" % json.dumps(json_res, ensure_ascii=False))
    return json.dumps(json_res, ensure_ascii=False)

if __name__ == "__main__":
    apps.run(debug=False, host='10.143.1.22')
