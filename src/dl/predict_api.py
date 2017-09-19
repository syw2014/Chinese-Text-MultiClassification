#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Jerry.Shi
# Date: 2017-08-15 18:02:30


from flask import Flask, render_template, request
from pyfasttext import FastText
import json

# load model
def load_model(pth):
    if len(pth) == 0:
        return None
    # create object
    model = FastText()
    model.load_model(pth)
    return model

model = load_model('../../result/model.bin')
# predcit with prob
def predict1(model, sentence):
    result = model.predict_proba_single(sentence)
    if len(result) == 0:
        json_res = {"sentence":sentence, "label": "null", "prob": "0.0"}
    else:
        json_res = {"sentence":sentence, "label": result[0][0], "prob": result[0][1]}
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
    apps.run(debug=True, host='10.143.1.22')
