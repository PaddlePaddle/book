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

def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  ch_in=None,
                  active_type=ReluActivation()):
    tmp = img_conv_layer(
        input=input,
        filter_size=filter_size,
        num_channels=ch_in,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=LinearActivation(),
        bias_attr=False)
    return batch_norm_layer(input=tmp, act=active_type)


def shortcut(ipt, n_in, n_out, stride):
    if n_in != n_out:
        return conv_bn_layer(ipt, n_out, 1, stride=stride, LinearActivation())
    else:
        return ipt

def basicblock(ipt, ch_out, stride):
    ch_in = ipt.num_filter
    tmp = conv_bn_layer(ipt, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, LinearActivation())
    short = shortcut(ipt, ch_in, ch_out, stride)
    return addto_layer(input=[input, short], act=ReluActivation())

def bottleneck(ipt, ch_out, stride):
    ch_in = ipt.num_filter
    tmp = conv_bn_layer(ipt, ch_out, 1, stride, 0)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1)
    tmp = conv_bn_layer(tmp, ch_out * 4, 1, 1, 0, LinearActivation())
    short = shortcut(ipt, ch_in, ch_out, stride)
    return addto_layer(input=[input, short], act=ReluActivation())

def layer_warp(block_func, ipt, features, count, stride):
    tmp = block_func(tmp, features, stride)
    for i in range(1, count):
        tmp = block_func(tmp, features, 1)
        return tmp

def resnet_imagenet(ipt, depth=50):
    cfg = {18 : ([2,2,2,1], basicblock),
           34 : ([3,4,6,3], basicblock),
           50 : ([3,4,6,3], bottleneck),
           101: ([3,4,23,3], bottleneck),
           152: ([3,8,36,3], bottleneck)}
    stages, block_func = cfg[depth]
    tmp = conv_bn_layer(ipt,
        ch_in=3,
        ch_out=64,
        filter_size=7,
        stride=2,
        padding=3)
    tmp = img_pool_layer(input=tmp, pool_size=3, stride=2)
    tmp = layer_warp(block_func, tmp,  64, stages[0], 1)
    tmp = layer_warp(block_func, tmp, 128, stages[1], 2)
    tmp = layer_warp(block_func, tmp, 256, stages[2], 2)
    tmp = layer_warp(block_func, tmp, 512, stages[3], 2)
    tmp = img_pool_layer(input=tmp,
                         pool_size=7,
                         stride=1,
                         pool_type=AvgPooling())

    tmp = fc_layer(input=tmp, size=1000, act=SoftmaxActivation())
    return tmp

def resnet_cifar10(ipt, depth=56):
    assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
    n = (depth - 2) / 6
    nStages = {16, 64, 128}
    tmp = conv_bn_layer(ipt,
        ch_in=3,
        ch_out=16,
        filter_size=3,
        stride=1,
        padding=1)
    tmp = layer_warp(basicblock, tmp, 16, n)
    tmp = layer_warp(basicblock, tmp, 32, n, 2)
    tmp = layer_warp(basicblock, tmp, 64, n, 2)
    tmp = img_pool_layer(input=tmp,
                         pool_size=8,
                         stride=1,
                         pool_type=AvgPooling())
    tmp = fc_layer(input=tmp, size=10, act=SoftmaxActivation())
    return tmp


is_predict = get_config_arg("is_predict", bool, False)
if not is_predict:
    args = {'meta': 'data/mean.meta'}
    define_py_data_sources2(
        train_list='data/train.list',
        test_list='data/test.list',
        module='dataprovider',
        obj='process',
        args=args)

settings(
    batch_size=128,
    learning_rate=0.1 / 128.0,
    learning_method=MomentumOptimizer(0.9),
    regularization=L2Regularization(0.0005 * 128))

data_size = 3 * 32 * 32
class_num = 10
data = data_layer(name='image', size=data_size)
out = resnet_cifar10(data, depth=50)
if not is_predict:
    lbl = data_layer(name="label", size=class_num)
    outputs(classification_cost(input=out, label=lbl))
else:
    outputs(out)
