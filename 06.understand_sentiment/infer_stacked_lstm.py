# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import os
import paddle
import paddle.fluid as fluid
from functools import partial
import numpy as np


CLASS_DIM = 2
EMB_DIM = 128
HID_DIM = 512
STACKED_NUM = 3
BATCH_SIZE = 128

def stacked_lstm_net(data, input_dim, class_dim, emb_dim, hid_dim, stacked_num):
    assert stacked_num % 2 == 1

    emb = fluid.layers.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)

    fc1 = fluid.layers.fc(input=emb, size=hid_dim)
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)

    inputs = [fc1, lstm1]

    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hid_dim)
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc, size=hid_dim, is_reverse=(i % 2) == 0)
        inputs = [fc, lstm]

    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

    prediction = fluid.layers.fc(
        input=[fc_last, lstm_last], size=class_dim, act='softmax')
    return prediction

def inference_program(word_dict):
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

    dict_dim = len(word_dict)
    net = stacked_lstm_net(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM,
                           STACKED_NUM)
    return net


def load_str_from_txt(file_path):
    if not os.path.isfile(file_path):
        raise TypeError(file_path + " dose not exist")

    comment = open(file_path).read()
    return comment


def infer(use_cuda, inference_program, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    word_dict = paddle.dataset.imdb.word_dict()

    inferencer = fluid.Inferencer(
        infer_func=partial(inference_program, word_dict),
        param_path=params_dirname,
        place=place)

    dir_path = "./test_samples/"
    test_samples_name = [
        '10897_10.txt',
        '11837_9.txt',
        '153_8.txt',
        '1547_9.txt',
        '5305_7.txt',
        '5674_1.txt',
        '6530_4.txt',
        '6539_2.txt',
        '7393_3.txt',
        '7403_2.txt' 
    ]
    reviews_str = [load_str_from_txt(dir_path + name) 
            for name in test_samples_name]
    reviews = [c.split() for c in reviews_str]

    UNK = word_dict['<unk>']
    lod = []
    for c in reviews:
        lod.append([word_dict.get(words, UNK) for words in c])

    base_shape = [[len(c) for c in lod]]

    tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
    results = inferencer.infer({'words': tensor_words})

    for i, r in enumerate(results[0]):
        print(test_samples_name[i], "positive: ", r[0], "negative: ", r[1])


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    params_dirname = "understand_sentiment_stacked_lstm.inference.model"
    infer(use_cuda, inference_program, params_dirname)


if __name__ == '__main__':
    use_cuda = False # set to True if training with GPU
    main(use_cuda)
