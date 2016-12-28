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

import math

#################### Data Configure ####################
args = {
    'srcText': 'data/simple-examples/data/ptb.train.txt',
    'dictfile': 'data/vocabulary.txt'
}
define_py_data_sources2(
    train_list="data/train.list",
    test_list="data/test.list",
    module="dataprovider",
    obj="process",
    args=args)

settings(
    batch_size=100, regularization=L2Regularization(8e-4), learning_rate=3e-3)

dictsize = 1953
embsize = 32
hiddensize = 256

firstword = data_layer(name="firstw", size=dictsize)
secondword = data_layer(name="secondw", size=dictsize)
thirdword = data_layer(name="thirdw", size=dictsize)
fourthword = data_layer(name="fourthw", size=dictsize)
nextword = data_layer(name="fifthw", size=dictsize)


# construct word embedding for each datalayer
def wordemb(inlayer):
    wordemb = table_projection(
        input=inlayer,
        size=embsize,
        param_attr=ParamAttr(
            name="_proj",
            initial_std=0.001,
            learning_rate=1,
            l2_rate=0, ))
    return wordemb


Efirst = wordemb(firstword)
Esecond = wordemb(secondword)
Ethird = wordemb(thirdword)
Efourth = wordemb(fourthword)

# concatentate Ngram embeddings into context embedding
contextemb = concat_layer(input=[Efirst, Esecond, Ethird, Efourth])
hidden1 = fc_layer(
    input=contextemb,
    size=hiddensize,
    act=SigmoidActivation(),
    layer_attr=ExtraAttr(drop_rate=0.5),
    bias_attr=ParamAttr(learning_rate=2),
    param_attr=ParamAttr(
        initial_std=1. / math.sqrt(embsize * 8), learning_rate=1))

# use context embedding to predict nextword
predictword = fc_layer(
    input=hidden1,
    size=dictsize,
    bias_attr=ParamAttr(learning_rate=2),
    act=SoftmaxActivation())

cost = classification_cost(input=predictword, label=nextword)

# network input and output
outputs(cost)
