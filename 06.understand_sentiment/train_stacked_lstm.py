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
import numpy as np
import sys
import math
import argparse

CLASS_DIM = 2
EMB_DIM = 128
HID_DIM = 512
STACKED_NUM = 3
BATCH_SIZE = 128


def parse_args():
    parser = argparse.ArgumentParser("stacked_lstm")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument(
        '--use_gpu', type=int, default=0, help="Whether to use GPU or not.")
    parser.add_argument(
        '--num_epochs', type=int, default=1, help="number of epochs.")
    args = parser.parse_args()
    return args


def stacked_lstm_net(data, input_dim, class_dim, emb_dim, hid_dim, stacked_num):
    assert stacked_num % 2 == 1

    emb = fluid.embedding(input=data, size=[input_dim, emb_dim], is_sparse=True)

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
    data = fluid.data(name="words", shape=[None], dtype="int64", lod_level=1)
    dict_dim = len(word_dict)
    net = stacked_lstm_net(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM,
                           STACKED_NUM)
    return net


def train_program(prediction):
    # prediction = inference_program(word_dict)
    label = fluid.data(name="label", shape=[None, 1], dtype="int64")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return [avg_cost, accuracy]


def optimizer_func():
    return fluid.optimizer.Adagrad(learning_rate=0.002)


def train(use_cuda, params_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    print("Loading IMDB word dict....")
    word_dict = paddle.dataset.imdb.word_dict()

    print("Reading training data....")

    if args.enable_ce:
        train_reader = paddle.batch(
            paddle.dataset.imdb.train(word_dict), batch_size=BATCH_SIZE)
    else:
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.imdb.train(word_dict), buf_size=25000),
            batch_size=BATCH_SIZE)

    print("Reading testing data....")
    test_reader = paddle.batch(
        paddle.dataset.imdb.test(word_dict), batch_size=BATCH_SIZE)

    feed_order = ['words', 'label']
    pass_num = args.num_epochs

    main_program = fluid.default_main_program()
    star_program = fluid.default_startup_program()

    if args.enable_ce:
        main_program.random_seed = 90
        star_program.random_seed = 90

    prediction = inference_program(word_dict)
    train_func_outputs = train_program(prediction)
    avg_cost = train_func_outputs[0]

    test_program = main_program.clone(for_test=True)

    # [avg_cost, accuracy] = train_program(prediction)
    sgd_optimizer = optimizer_func()
    sgd_optimizer.minimize(avg_cost)
    exe = fluid.Executor(place)

    def train_test(program, reader):
        count = 0
        feed_var_list = [
            program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
        test_exe = fluid.Executor(place)
        accumulated = len(train_func_outputs) * [0]
        for test_data in reader():
            avg_cost_np = test_exe.run(
                program=program,
                feed=feeder_test.feed(test_data),
                fetch_list=train_func_outputs)
            accumulated = [
                x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)
            ]
            count += 1
        return [x / count for x in accumulated]

    def train_loop():

        feed_var_list_loop = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
        exe.run(star_program)

        for epoch_id in range(pass_num):
            for step_id, data in enumerate(train_reader()):
                metrics = exe.run(
                    main_program,
                    feed=feeder.feed(data),
                    fetch_list=[var.name for var in train_func_outputs])
                print("step: {0}, Metrics {1}".format(
                    step_id, list(map(np.array, metrics))))
                if (step_id + 1) % 10 == 0:
                    avg_cost_test, acc_test = train_test(test_program,
                                                         test_reader)
                    print('Step {0}, Test Loss {1:0.2}, Acc {2:0.2}'.format(
                        step_id, avg_cost_test, acc_test))

                    print("Step {0}, Epoch {1} Metrics {2}".format(
                        step_id, epoch_id, list(map(np.array, metrics))))
                if math.isnan(float(metrics[0])):
                    sys.exit("got NaN loss, training failed.")
            if params_dirname is not None:
                fluid.io.save_inference_model(params_dirname, ["words"],
                                              prediction, exe)
            if args.enable_ce and epoch_id == pass_num - 1:
                print("kpis\tlstm_train_cost\t%f" % metrics[0])
                print("kpis\tlstm_train_acc\t%f" % metrics[1])
                print("kpis\tlstm_test_cost\t%f" % avg_cost_test)
                print("kpis\tlstm_test_acc\t%f" % acc_test)

    train_loop()


def infer(use_cuda, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    word_dict = paddle.dataset.imdb.word_dict()

    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inferencer, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

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
            lod.append([np.int64(word_dict.get(words, UNK)) for words in c])

        base_shape = [[len(c) for c in lod]]

        tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
        assert feed_target_names[0] == "words"
        results = exe.run(
            inferencer,
            feed={feed_target_names[0]: tensor_words},
            fetch_list=fetch_targets,
            return_numpy=False)
        np_data = np.array(results[0])
        for i, r in enumerate(np_data):
            print("Predict probability of ", r[0], " to be positive and ", r[1],
                  " to be negative for review \'", reviews_str[i], "\'")


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    params_dirname = "understand_sentiment_stacked_lstm.inference.model"
    train(use_cuda, params_dirname)
    infer(use_cuda, params_dirname)


if __name__ == '__main__':
    args = parse_args()
    use_cuda = args.use_gpu  # set to True if training with GPU
    main(use_cuda)
