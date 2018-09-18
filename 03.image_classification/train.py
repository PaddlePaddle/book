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
# limitations under the License

from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy
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

from vgg import vgg_bn_drop
from resnet import resnet_cifar10


def inference_network():
    # The image is 32 * 32 with RGB representation.
    data_shape = [3, 32, 32]
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')

    predict = resnet_cifar10(images, 32)
    # predict = vgg_bn_drop(images) # un-comment to use vgg net
    return predict


def train_network():
    predict = inference_network()
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, accuracy]


def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)


def train(use_cuda, train_program, params_dirname):
    BATCH_SIZE = 128
    EPOCH_NUM = 2

    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.cifar.train10(), buf_size=50000),
        batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)

    def event_handler(event):
        if isinstance(event, EndStepEvent):
            if event.step % 100 == 0:
                print("\nPass %d, Batch %d, Cost %f, Acc %f" %
                      (event.step, event.epoch, event.metrics[0],
                       event.metrics[1]))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

        if isinstance(event, EndEpochEvent):
            avg_cost, accuracy = trainer.test(
                reader=test_reader, feed_order=['pixel', 'label'])

            print('\nTest with Pass {0}, Loss {1:2.2}, Acc {2:2.2}'.format(
                event.epoch, avg_cost, accuracy))
            if params_dirname is not None:
                trainer.save_params(params_dirname)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    trainer = Trainer(
        train_func=train_program, optimizer_func=optimizer_program, place=place)

    trainer.train(
        reader=train_reader,
        num_epochs=EPOCH_NUM,
        event_handler=event_handler,
        feed_order=['pixel', 'label'])


def infer(use_cuda, inference_program, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inferencer = Inferencer(
        infer_func=inference_program, param_path=params_dirname, place=place)

    # Prepare testing data. 
    from PIL import Image
    import numpy as np
    import os

    def load_image(file):
        im = Image.open(file)
        im = im.resize((32, 32), Image.ANTIALIAS)

        im = np.array(im).astype(np.float32)
        # The storage order of the loaded image is W(width),
        # H(height), C(channel). PaddlePaddle requires
        # the CHW order, so transpose them.
        im = im.transpose((2, 0, 1))  # CHW
        im = im / 255.0

        # Add one dimension to mimic the list format.
        im = numpy.expand_dims(im, axis=0)
        return im

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img = load_image(cur_dir + '/image/dog.png')

    # inference
    results = inferencer.infer({'pixel': img})

    label_list = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
        "ship", "truck"
    ]
    print("infer results: %s" % label_list[np.argmax(results[0])])


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    save_path = "image_classification_resnet.inference.model"

    train(
        use_cuda=use_cuda,
        train_program=train_network,
        params_dirname=save_path)

    infer(
        use_cuda=use_cuda,
        inference_program=inference_network,
        params_dirname=save_path)


if __name__ == '__main__':
    # For demo purpose, the training runs on CPU
    # Please change accordingly.
    main(use_cuda=False)
