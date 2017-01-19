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

from paddle.trainer_config_helpers import *

is_predict = get_config_arg("is_predict", bool, False)

####################Data Configuration ##################

if not is_predict:
    data_dir = './data/'
    define_py_data_sources2(
        train_list=data_dir + 'train.list',
        test_list=data_dir + 'test.list',
        module='mnist_provider',
        obj='process')

######################Algorithm Configuration #############
settings(
    batch_size=128,
    learning_rate=0.1 / 128.0,
    learning_method=MomentumOptimizer(0.9),
    regularization=L2Regularization(0.0005 * 128))

#######################Network Configuration #############

data_size = 1 * 28 * 28
label_size = 10
img = data_layer(name='pixel', size=data_size)


def softmax_regression(img):
    predict = fc_layer(input=img, size=10, act=SoftmaxActivation())
    return predict


def multilayer_perceptron(img):
    # The first fully-connected layer
    hidden1 = fc_layer(input=img, size=128, act=ReluActivation())
    # The second fully-connected layer and the according activation function
    hidden2 = fc_layer(input=hidden1, size=64, act=ReluActivation())
    # The thrid fully-connected layer, note that the hidden size should be 10,
    # which is the number of unique digits
    predict = fc_layer(input=hidden2, size=10, act=SoftmaxActivation())
    return predict


def convolutional_neural_network(img):
    # first conv layer
    conv_pool_1 = simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=TanhActivation())
    # second conv layer
    conv_pool_2 = simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=TanhActivation())
    # The first fully-connected layer
    fc1 = fc_layer(input=conv_pool_2, size=128, act=TanhActivation())
    # The softmax layer, note that the hidden size should be 10,
    # which is the number of unique digits
    predict = fc_layer(input=fc1, size=10, act=SoftmaxActivation())
    return predict


predict = softmax_regression(img)
#predict = multilayer_perceptron(img)
#predict = convolutional_neural_network(img)

if not is_predict:
    lbl = data_layer(name="label", size=label_size)
    inputs(img, lbl)
    outputs(classification_cost(input=predict, label=lbl))
else:
    outputs(predict)
