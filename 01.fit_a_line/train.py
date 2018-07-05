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

import paddle
import paddle.fluid as fluid
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

trainer = fluid.Trainer(
    train_func=train_program, place=place, optimizer_func=optimizer_program)

feed_order = ['x', 'y']

# Specify the directory path to save the parameters
params_dirname = "fit_a_line.inference.model"

# Plot data
from paddle.v2.plot import Ploter
train_title = "Train cost"
test_title = "Test cost"
plot_cost = Ploter(train_title, test_title)

step = 0


# event_handler to print training and testing info
def event_handler_plot(event):
    global step
    if isinstance(event, fluid.EndStepEvent):
        if event.step % 10 == 0:  # every 10 batches, record a test cost
            test_metrics = trainer.test(
                reader=test_reader, feed_order=feed_order)

            plot_cost.append(test_title, step, test_metrics[0])
            plot_cost.plot()

            if test_metrics[0] < 10.0:
                # If the accuracy is good enough, we can stop the training.
                print('loss is less than 10.0, stop')
                trainer.stop()

        # We can save the trained parameters for the inferences later
        if params_dirname is not None:
            trainer.save_params(params_dirname)

        step += 1


# The training could take up to a few minutes.
trainer.train(
    reader=train_reader,
    num_epochs=100,
    event_handler=event_handler_plot,
    feed_order=feed_order)


def inference_program():
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    return y_predict


inferencer = fluid.Inferencer(
    infer_func=inference_program, param_path=params_dirname, place=place)

batch_size = 10
tensor_x = numpy.random.uniform(0, 10, [batch_size, 13]).astype("float32")

results = inferencer.infer({'x': tensor_x})
print("infer results: ", results[0])
