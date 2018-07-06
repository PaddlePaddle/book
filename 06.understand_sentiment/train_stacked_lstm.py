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
USE_GPU = False


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


def train_program(word_dict):
    prediction = inference_program(word_dict)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return [avg_cost, accuracy]


def optimizer_func():
    return fluid.optimizer.Adagrad(learning_rate=0.002)


def train(use_cuda, train_program, params_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    print("Loading IMDB word dict....")
    word_dict = paddle.dataset.imdb.word_dict()

    print("Reading training data....")
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=25000),
        batch_size=BATCH_SIZE)

    print("Reading testing data....")
    test_reader = paddle.batch(
        paddle.dataset.imdb.test(word_dict), batch_size=BATCH_SIZE)

    trainer = fluid.Trainer(
        train_func=partial(train_program, word_dict),
        place=place,
        optimizer_func=optimizer_func)

    feed_order = ['words', 'label']

    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            if event.step % 10 == 0:
                avg_cost, acc = trainer.test(
                    reader=test_reader, feed_order=feed_order)

                print('Step {0}, Test Loss {1:0.2}, Acc {2:0.2}'.format(
                    event.step, avg_cost, acc))

                print("Step {0}, Epoch {1} Metrics {2}".format(
                    event.step, event.epoch, map(np.array, event.metrics)))

        elif isinstance(event, fluid.EndEpochEvent):
            trainer.save_params(params_dirname)

    trainer.train(
        num_epochs=1,
        event_handler=event_handler,
        reader=train_reader,
        feed_order=feed_order)


def infer(use_cuda, inference_program, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    word_dict = paddle.dataset.imdb.word_dict()

    inferencer = fluid.Inferencer(
        infer_func=partial(inference_program, word_dict),
        param_path=params_dirname,
        place=place)

    # Setup input by creating LoDTensor to represent sequence of words.
    # Here each word is the basic element of the LoDTensor and the shape of 
    # each word (base_shape) should be [1] since it is simply an index to 
    # look up for the corresponding word vector.
    # Suppose the length_based level of detail (lod) info is set to [[3, 4, 2]],
    # which has only one lod level. Then the created LoDTensor will have only 
    # one higher level structure (sequence of words, or sentence) than the basic 
    # element (word). Hence the LoDTensor will hold data for three sentences of 
    # length 3, 4 and 2, respectively. 
    # Note that lod info should be a list of lists.

    reviews_str = [
        'read the book forget the movie', 'this is a great movie',
        'this is very bad'
    ]
    reviews = [c.split() for c in reviews_str]

    UNK = word_dict['<unk>']
    lod = []
    for c in reviews:
        lod.append([word_dict.get(words, UNK) for words in c])

    base_shape = [[len(c) for c in lod]]

    tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
    results = inferencer.infer({'words': tensor_words})

    for i, r in enumerate(results[0]):
        print("Predict probability of ", r[0], " to be positive and ", r[1],
              " to be negative for review \'", reviews_str[i], "\'")


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    params_dirname = "understand_sentiment_stacked_lstm.inference.model"
    train(use_cuda, train_program, params_dirname)
    infer(use_cuda, inference_program, params_dirname)


if __name__ == '__main__':
    use_cuda = False # set to True if training with GPU
    main(use_cuda)
