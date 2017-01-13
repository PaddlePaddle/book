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
from collections import Counter
from urllib2 import urlopen
import argparse
import os
import random
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
data_url = 'https://archive.ics.uci.edu/ml/machine' \
           '-learning-databases/housing/housing.data'
raw_data = 'housing.data'
train_data = 'housing.train.npy'
test_data = 'housing.test.npy'
feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT'
]
root_dir = os.path.abspath(os.pardir)


def maybe_download(url, file_path):
    if not os.path.exists(file_path):
        logging.info('data doesn\'t exist on %s, download from [%s]' %
                     (file_path, url))
        resp = urlopen(url).read()
        with open(file_path, 'w') as f:
            f.write(resp)

    logging.info('got raw housing data')


def save_list():
    with open('train.list', 'w') as f:
        f.write('data/' + train_data + '\n')
    with open('test.list', 'w') as f:
        f.write('data/' + test_data + '\n')


def feature_range(maximums, minimums):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    feature_num = len(maximums)
    ax.bar(range(feature_num), maximums - minimums, color='r', align='center')
    ax.set_title('feature scale')
    plt.xticks(range(feature_num), feature_names)
    plt.xlim([-1, feature_num])
    fig.set_figheight(6)
    fig.set_figwidth(10)
    fig.savefig('%s/image/ranges.png' % root_dir, dpi=48)
    plt.close(fig)


def preprocess(file_path, feature_num=14, shuffle=False, ratio=0.8):
    data = np.fromfile(file_path, sep=' ')
    data = data.reshape(data.shape[0] / feature_num, feature_num)
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(
        axis=0) / data.shape[0]
    feature_range(maximums[:-1], minimums[:-1])
    for i in xrange(feature_num - 1):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    if shuffle:
        np.random.shuffle(data)
    offset = int(data.shape[0] * ratio)
    np.save(train_data, data[:offset])
    logging.info('saved training data to %s' % train_data)
    np.save(test_data, data[offset:])
    logging.info('saved test data to %s' % test_data)
    save_list()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='download boston housing price data set and preprocess the data(normalization and split dataset)'
    )
    parser.add_argument(
        '-r',
        '--ratio',
        dest='ratio',
        default='0.8',
        help='ratio of data used for training')
    parser.add_argument(
        '-s',
        '--shuffle',
        dest='shuffle',
        default='0',
        choices={'1', '0'},
        help='shuffle the data before splitting, 1=shuffle, 0=do not shuffle')
    args = parser.parse_args()

    maybe_download(data_url, raw_data)
    preprocess(raw_data, shuffle=int(args.shuffle), ratio=float(args.ratio))
