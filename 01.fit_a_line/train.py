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

import paddle
import paddle.fluid as fluid
import numpy
import math
import sys


def main():

    batch_size = 20
    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.uci_housing.test(), buf_size=500),
        batch_size=batch_size)

    # feature vector of length 13
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    main_program = fluid.default_main_program()
    star_program = fluid.default_startup_program()

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_loss)

    test_program = main_program.clone(for_test=True)

    # can use CPU or GPU
    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Specify the directory to save the parameters
    params_dirname = "fit_a_line.inference.model"
    num_epochs = 100

    # For training test cost
    def train_test(program, feeder):
        exe_test = fluid.Executor(place)
        accumulated = 1 * [0]
        count = 0
        for data_test in test_reader():
            outs = exe_test.run(
                program=program,
                feed=feeder.feed(data_test),
                fetch_list=[avg_loss])
            accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]
            count += 1
        return [x_d / count for x_d in accumulated]

    # main train loop.
    def train_loop():

        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        feeder_test = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe.run(star_program)

        train_title = "Train cost"
        test_title = "Test cost"
        step = 0

        for pass_id in range(num_epochs):
            for data_train in train_reader():
                avg_loss_value, = exe.run(
                    main_program,
                    feed=feeder.feed(data_train),
                    fetch_list=[avg_loss])
                if step % 10 == 0:  # record a train cost every 10 batches
                    print("%s, Step %d, Cost %f" %
                          (train_title, step, avg_loss_value[0]))
                if step % 100 == 0:  # record a test cost every 100 batches
                    test_metics = train_test(
                        program=test_program, feeder=feeder_test)
                    print("%s, Step %d, Cost %f" %
                          (test_title, step, test_metics[0]))
                    # If the accuracy is good enough, we can stop the training.
                    if test_metics[0] < 10.0:
                        return

                step += 1

                if math.isnan(float(avg_loss_value)):
                    sys.exit("got NaN loss, training failed.")
            if params_dirname is not None:
                # We can save the trained parameters for the inferences later
                fluid.io.save_inference_model(params_dirname, ['x'],
                                              [y_predict], exe)

    train_loop()

    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    # infer
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)
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


if __name__ == '__main__':
    main()
