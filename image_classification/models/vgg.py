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

def vgg_bn_drop(input, num_channels):
    def conv_block(ipt, num_filter, groups, dropouts, num_channels_=None):
        return img_conv_group(
            input=ipt,
            num_channels=num_channels_,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=ReluActivation(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=MaxPooling())

    tmp = conv_block(input, 64, 2, [0.3, 0], num_channels)
    tmp = conv_block(tmp, 128, 2, [0.4, 0])
    tmp = conv_block(tmp, 256, 3, [0.4, 0.4, 0])
    tmp = conv_block(tmp, 512, 3, [0.4, 0.4, 0])
    tmp = conv_block(tmp, 512, 3, [0.4, 0.4, 0])

    tmp = dropout_layer(input=tmp, dropout_rate=0.5)
    tmp = fc_layer(
        input=tmp,
        size=512,
        act=LinearActivation())
    tmp = batch_norm_layer(input=tmp,
        act=ReluActivation(),
        layer_attr=ExtraAttr(drop_rate=0.5))
    tmp = fc_layer(
        input=tmp,
        size=512,
        act=LinearActivation())
    tmp = fc_layer(input=tmp, size=10, act=SoftmaxActivation())
    return tmp

is_predict = get_config_arg("is_predict", bool, False)
if not is_predict:
    define_py_data_sources2(
        train_list='data/train.list',
        test_list='data/test.list',
        module='dataprovider',
        obj='process',
        args={'mean_path': 'data/mean.meta'})

settings(
    batch_size=128,
    learning_rate=0.1 / 128.0,
    learning_rate_decay_a=0.1,
    learning_rate_decay_b=50000 * 100,
    learning_rate_schedule='discexp',
    learning_method=MomentumOptimizer(0.9),
    regularization=L2Regularization(0.0005 * 128),)

data_size = 3 * 32 * 32
class_num = 10
data = data_layer(name='image', size=data_size)
out = vgg_bn_drop(data, 3)
if not is_predict:
    lbl = data_layer(name="label", size=class_num)
    outputs(classification_cost(input=out, label=lbl))
else:
    outputs(out)
