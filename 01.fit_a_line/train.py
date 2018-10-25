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
import sys

try:
    from paddle.fluid.contrib.trainer import *
    from paddle.fluid.contrib.inferencer import *
except ImportError:
    print(
        "In the fluid 1.0, the trainer and inferencer are moving to paddle.fluid.contrib",
        file=sys.stderr)
    from paddle.fluid.trainer import *
    from paddle.fluid.inferencer import *

import numpy

BATCH_SIZE = 20

train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.train(), buf_size=500),
    batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.test(), buf_size=500),
    batch_size=BATCH_SIZE)


def train_program():
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    # feature vector of length 13
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    loss = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(loss)

    return avg_loss


def optimizer_program():
    return fluid.optimizer.SGD(learning_rate=0.001)


# can use CPU or GPU
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

trainer = Trainer(
    train_func=train_program, place=place, optimizer_func=optimizer_program)

feed_order = ['x', 'y']

# Specify the directory to save the parameters
params_dirname = "fit_a_line.inference.model"

train_title = "Train cost"
test_title = "Test cost"

step = 0


# event_handler prints training and testing info
def event_handler(event):
    global step
    if isinstance(event, EndStepEvent):
        if step % 10 == 0:  # record a train cost every 10 batches
            print("%s, Step %d, Cost %f" %
                  (train_title, step, event.metrics[0]))
        if step % 100 == 0:  # record a test cost every 100 batches
            test_metrics = trainer.test(
                reader=test_reader, feed_order=feed_order)
            print("%s, Step %d, Cost %f" % (test_title, step, test_metrics[0]))
            if test_metrics[0] < 10.0:
                # If the accuracy is good enough, we can stop the training.
                print('loss is less than 10.0, stop')
                trainer.stop()
        step += 1

    if isinstance(event, EndEpochEvent):
        if event.epoch % 10 == 0:
            # We can save the trained parameters for the inferences later
            if params_dirname is not None:
                trainer.save_params(params_dirname)


# The training could take up to a few minutes.
trainer.train(
    reader=train_reader,
    num_epochs=100,
    event_handler=event_handler,
    feed_order=feed_order)


def inference_program():
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    return y_predict


inferencer = Inferencer(
    infer_func=inference_program, param_path=params_dirname, place=place)

batch_size = 10
test_reader = paddle.batch(
    paddle.dataset.uci_housing.test(), batch_size=batch_size)
test_data = next(test_reader())
test_x = numpy.array([data[0] for data in test_data]).astype("float32")
test_y = numpy.array([data[1] for data in test_data]).astype("float32")

results = inferencer.infer({'x': test_x})

print("infer results: (House Price)")
for idx, val in enumerate(results[0]):
    print("%d: %.2f" % (idx, val))

print("\nground truth:")
for idx, val in enumerate(test_y):
    print("%d: %.2f" % (idx, val))
