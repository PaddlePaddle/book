#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import argparse

import math
import numpy

import paddle
import paddle.fluid as fluid


def parse_args():
    parser = argparse.ArgumentParser("fit_a_line")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=False,
        help="Whether to use GPU or not.")
    parser.add_argument(
        '--num_epochs', type=int, default=100, help="number of epochs.")
    args = parser.parse_args()
    return args


# For training test cost
def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(
            program=program, feed=feeder.feed(data_test), fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]
        count += 1
    return [x_d / count for x_d in accumulated]


def save_result(points1, points2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x1 = [idx for idx in range(len(points1))]
    y1 = points1
    y2 = points2
    l1 = plt.plot(x1, y1, 'r--', label='predictions')
    l2 = plt.plot(x1, y2, 'g--', label='GT')
    plt.plot(x1, y1, 'ro-', x1, y2, 'g+-')
    plt.title('predictions VS GT')
    plt.legend()
    plt.savefig('./image/prediction_gt.png')


def main():
    batch_size = 20

    if args.enable_ce:
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.dataset.uci_housing.test(), batch_size=batch_size)
    else:
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.uci_housing.train(), buf_size=500),
            batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.uci_housing.test(), buf_size=500),
            batch_size=batch_size)

    # feature vector of length 13
    x = fluid.data(name='x', shape=[None, 13], dtype='float32')
    y = fluid.data(name='y', shape=[None, 1], dtype='float32')

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    if args.enable_ce:
        main_program.random_seed = 90
        startup_program.random_seed = 90

    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)

    test_program = main_program.clone(for_test=True)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_loss)

    # can use CPU or GPU
    use_cuda = args.use_gpu
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Specify the directory to save the parameters
    params_dirname = "fit_a_line.inference.model"
    num_epochs = args.num_epochs

    # main train loop.
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    exe.run(startup_program)

    train_prompt = "Train cost"
    test_prompt = "Test cost"
    step = 0

    exe_test = fluid.Executor(place)

    for pass_id in range(num_epochs):
        for data_train in train_reader():
            avg_loss_value, = exe.run(
                main_program,
                feed=feeder.feed(data_train),
                fetch_list=[avg_loss])
            if step % 10 == 0:  # record a train cost every 10 batches
                print("%s, Step %d, Cost %f" %
                      (train_prompt, step, avg_loss_value[0]))

            if step % 100 == 0:  # record a test cost every 100 batches
                test_metics = train_test(
                    executor=exe_test,
                    program=test_program,
                    reader=test_reader,
                    fetch_list=[avg_loss],
                    feeder=feeder)
                print("%s, Step %d, Cost %f" %
                      (test_prompt, step, test_metics[0]))
                # If the accuracy is good enough, we can stop the training.
                if test_metics[0] < 10.0:
                    break

            step += 1

            if math.isnan(float(avg_loss_value[0])):
                sys.exit("got NaN loss, training failed.")
        if params_dirname is not None:
            # We can save the trained parameters for the inferences later
            fluid.io.save_inference_model(params_dirname, ['x'], [y_predict],
                                          exe)

        if args.enable_ce and pass_id == args.num_epochs - 1:
            print("kpis\ttrain_cost\t%f" % avg_loss_value[0])
            print("kpis\ttest_cost\t%f" % test_metics[0])

    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    # infer
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets
         ] = fluid.io.load_inference_model(params_dirname, infer_exe)
        batch_size = 10

        infer_reader = paddle.batch(
            paddle.dataset.uci_housing.test(), batch_size=batch_size)

        infer_data = next(infer_reader())
        infer_feat = numpy.array(
            [data[0] for data in infer_data]).astype("float32")
        infer_label = numpy.array(
            [data[1] for data in infer_data]).astype("float32")

        assert feed_target_names[0] == 'x'
        results = infer_exe.run(
            inference_program,
            feed={feed_target_names[0]: numpy.array(infer_feat)},
            fetch_list=fetch_targets)

        print("infer results: (House Price)")
        for idx, val in enumerate(results[0]):
            print("%d: %.2f" % (idx, val))

        print("\nground truth:")
        for idx, val in enumerate(infer_label):
            print("%d: %.2f" % (idx, val))

        save_result(results[0], infer_label)


if __name__ == '__main__':
    args = parse_args()
    main()
