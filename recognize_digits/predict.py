# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Usage: predict.py -c CONF -d ./data/raw_data/  -m MODEL


Arguments:
    CONF        train conf
    DATA        MNIST Data
    MODEL       Model

Options:
    -h      --help
    -c      conf
    -d      data
    -m      model
"""

import os
import sys
from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np

from py_paddle import swig_paddle, DataProviderConverter
from paddle.trainer.PyDataProvider2 import dense_vector
from paddle.trainer.config_parser import parse_config

from load_data import read_data


class Prediction():
    def __init__(self, train_conf, data_dir, model_dir):

        conf = parse_config(train_conf, 'is_predict=1')
        self.network = swig_paddle.GradientMachine.createFromConfigProto(
            conf.model_config)
        self.network.loadParameters(model_dir)

        self.images, self.labels = read_data(data_dir, "t10k")

        slots = [dense_vector(28 * 28)]
        self.converter = DataProviderConverter(slots)

    def predict(self, index):
        input = self.converter([[self.images[index].flatten().tolist()]])
        output = self.network.forwardTest(input)
        prob = output[0]["value"]
        predict = np.argsort(-prob)
        print "Predicted probability of each digit:"
        print prob
        print "Predict Number: %d" % predict[0][0]
        print "Actual Number: %d" % self.labels[index]


def main():
    arguments = docopt(__doc__)
    train_conf = arguments['CONF']
    data_dir = arguments['DATA']
    model_dir = arguments['MODEL']
    swig_paddle.initPaddle("--use_gpu=0")
    predictor = Prediction(train_conf, data_dir, model_dir)
    while True:
        index = int(raw_input("Input image_id [0~9999]: "))
        predictor.predict(index)


if __name__ == '__main__':
    main()
