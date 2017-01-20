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
import logging
import argparse
import numpy as np
from py_paddle import swig_paddle, DataProviderConverter
from paddle.trainer.PyDataProvider2 import *
from paddle.trainer.config_parser import parse_config

logging.basicConfig(level=logging.INFO)


def predict(input_file, model_dir):
    # prepare PaddlePaddle environment, load models
    swig_paddle.initPaddle("--use_gpu=0")
    conf = parse_config('trainer_config.py', 'is_predict=1')
    network = swig_paddle.GradientMachine.createFromConfigProto(
        conf.model_config)
    network.loadParameters(model_dir)
    slots = [dense_vector(13)]
    converter = DataProviderConverter(slots)

    data = np.load(input_file)
    ys = []
    for row in data:
        result = network.forwardTest(converter([[row[:-1].tolist()]]))
        y_true = row[-1:].tolist()[0]
        y_predict = result[0]['value'][0][0]
        ys.append([y_true, y_predict])

    ys = np.matrix(ys)
    avg_err = np.average(np.square((ys[:, 0] - ys[:, 1])))
    logging.info('MSE of test set is %f' % avg_err)

    # draw a scatter plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.scatter(ys[:, 0], ys[:, 1])
    y_range = [ys[:, 0].min(), ys[:, 0].max()]
    ax.plot(y_range, y_range, 'k--', lw=4)
    ax.set_xlabel('True ($1000)')
    ax.set_ylabel('Predicted ($1000)')
    ax.set_title('Predictions on boston housing price')
    fig.savefig('image/predictions.png', dpi=60)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='predict house price and save the result as image.')
    parser.add_argument(
        '-m',
        '--model',
        dest='model',
        default='output/pass-00029',
        help='model path')
    parser.add_argument(
        '-t',
        '--test_data',
        dest='test_data',
        default='data/housing.test.npy',
        help='test data path')
    args = parser.parse_args()

    predict(input_file=args.test_data, model_dir=args.model)
