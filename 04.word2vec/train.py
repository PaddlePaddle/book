#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import paddle as paddle
import paddle.fluid as fluid
import six
import numpy
import sys
import math
import argparse

EMBED_SIZE = 32
HIDDEN_SIZE = 256
N = 5
BATCH_SIZE = 100

word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)


def parse_args():
    parser = argparse.ArgumentParser("word2vec")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help='If set, run the task with continuous evaluation logs.')
    parser.add_argument(
        '--use_gpu', type=int, default=0, help='whether to use gpu')
    parser.add_argument(
        '--num_epochs', type=int, default=100, help='number of epoch')
    args = parser.parse_args()
    return args


def inference_program(words, is_sparse):

    embed_first = fluid.embedding(
        input=words[0],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_second = fluid.embedding(
        input=words[1],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_third = fluid.embedding(
        input=words[2],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_fourth = fluid.embedding(
        input=words[3],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')

    concat_embed = fluid.layers.concat(
        input=[embed_first, embed_second, embed_third, embed_fourth], axis=1)
    hidden1 = fluid.layers.fc(
        input=concat_embed, size=HIDDEN_SIZE, act='sigmoid')
    predict_word = fluid.layers.fc(input=hidden1, size=dict_size, act='softmax')
    return predict_word


def train_program(predict_word):
    # The declaration of 'next_word' must be after the invoking of inference_program,
    # or the data input order of train program would be [next_word, firstw, secondw,
    # thirdw, fourthw], which is not correct.
    next_word = fluid.data(name='nextw', shape=[None, 1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def optimizer_func():
    return fluid.optimizer.AdagradOptimizer(
        learning_rate=3e-3,
        regularization=fluid.regularizer.L2DecayRegularizer(8e-4))


def train(if_use_cuda, params_dirname, is_sparse=True):
    place = fluid.CUDAPlace(0) if if_use_cuda else fluid.CPUPlace()

    train_reader = paddle.batch(
        paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.imikolov.test(word_dict, N), BATCH_SIZE)

    first_word = fluid.data(name='firstw', shape=[None, 1], dtype='int64')
    second_word = fluid.data(name='secondw', shape=[None, 1], dtype='int64')
    third_word = fluid.data(name='thirdw', shape=[None, 1], dtype='int64')
    forth_word = fluid.data(name='fourthw', shape=[None, 1], dtype='int64')
    next_word = fluid.data(name='nextw', shape=[None, 1], dtype='int64')

    word_list = [first_word, second_word, third_word, forth_word, next_word]
    feed_order = ['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw']

    main_program = fluid.default_main_program()
    star_program = fluid.default_startup_program()

    if args.enable_ce:
        main_program.random_seed = 90
        star_program.random_seed = 90

    predict_word = inference_program(word_list, is_sparse)
    avg_cost = train_program(predict_word)
    test_program = main_program.clone(for_test=True)

    optimizer = optimizer_func()
    optimizer.minimize(avg_cost)

    exe = fluid.Executor(place)

    def train_test(program, reader):
        count = 0
        feed_var_list = [
            program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
        test_exe = fluid.Executor(place)
        accumulated = len([avg_cost]) * [0]
        for test_data in reader():
            avg_cost_np = test_exe.run(
                program=program,
                feed=feeder_test.feed(test_data),
                fetch_list=[avg_cost])
            accumulated = [
                x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)
            ]
            count += 1
        return [x / count for x in accumulated]

    def train_loop():
        step = 0
        feed_var_list_loop = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
        exe.run(star_program)
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                avg_cost_np = exe.run(
                    main_program, feed=feeder.feed(data), fetch_list=[avg_cost])

                if step % 10 == 0:
                    outs = train_test(test_program, test_reader)
                    # print("Step %d: Average Cost %f" % (step, avg_cost_np[0]))
                    print("Step %d: Average Cost %f" % (step, outs[0]))
                    # print(outs)

                    # it will take a few hours.
                    # If average cost is lower than 5.8, we consider the model good enough to stop.
                    # Note 5.8 is a relatively high value. In order to get a better model, one should
                    # aim for avg_cost lower than 3.5. But the training could take longer time.
                    if outs[0] < 5.8:
                        if args.enable_ce:
                            print("kpis\ttrain_cost\t%f" % outs[0])

                        if params_dirname is not None:
                            fluid.io.save_inference_model(params_dirname, [
                                'firstw', 'secondw', 'thirdw', 'fourthw'
                            ], [predict_word], exe)
                        return
                step += 1
                if math.isnan(float(avg_cost_np[0])):
                    sys.exit("got NaN loss, training failed.")
        raise AssertionError("Cost is too large {0:2.2}".format(avg_cost_np[0]))

    train_loop()


def infer(use_cuda, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inferencer, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

        # Setup inputs by creating 4 LoDTensors representing 4 words. Here each word
        # is simply an index to look up for the corresponding word vector and hence
        # the shape of word (base_shape) should be [1]. The recursive_sequence_lengths,
        # which is length-based level of detail (lod) of each LoDTensor, should be [[1]]
        # meaning there is only one level of detail and there is only one sequence of
        # one word on this level.
        # Note that recursive_sequence_lengths should be a list of lists.
        data1 = numpy.asarray([[211]], dtype=numpy.int64)  # 'among'
        data2 = numpy.asarray([[6]], dtype=numpy.int64)  # 'a'
        data3 = numpy.asarray([[96]], dtype=numpy.int64)  # 'group'
        data4 = numpy.asarray([[4]], dtype=numpy.int64)  # 'of'
        lod = numpy.asarray([[1]], dtype=numpy.int64)

        first_word = fluid.create_lod_tensor(data1, lod, place)
        second_word = fluid.create_lod_tensor(data2, lod, place)
        third_word = fluid.create_lod_tensor(data3, lod, place)
        fourth_word = fluid.create_lod_tensor(data4, lod, place)

        assert feed_target_names[0] == 'firstw'
        assert feed_target_names[1] == 'secondw'
        assert feed_target_names[2] == 'thirdw'
        assert feed_target_names[3] == 'fourthw'

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        results = exe.run(
            inferencer,
            feed={
                feed_target_names[0]: first_word,
                feed_target_names[1]: second_word,
                feed_target_names[2]: third_word,
                feed_target_names[3]: fourth_word
            },
            fetch_list=fetch_targets,
            return_numpy=False)

        print(numpy.array(results[0]))
        most_possible_word_index = numpy.argmax(results[0])
        print(most_possible_word_index)
        print([
            key for key, value in six.iteritems(word_dict)
            if value == most_possible_word_index
        ][0])

        print(results[0].recursive_sequence_lengths())
        np_data = numpy.array(results[0])
        print("Inference Shape: ", np_data.shape)


def main(use_cuda, is_sparse):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    params_dirname = "word2vec.inference.model"

    train(
        if_use_cuda=use_cuda,
        params_dirname=params_dirname,
        is_sparse=is_sparse)

    infer(use_cuda=use_cuda, params_dirname=params_dirname)


if __name__ == '__main__':
    args = parse_args()
    PASS_NUM = args.num_epochs
    use_cuda = args.use_gpu  # set to True if training with GPU
    main(use_cuda=use_cuda, is_sparse=True)
