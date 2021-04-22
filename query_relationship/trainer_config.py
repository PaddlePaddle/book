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

is_predict = get_config_arg('is_predict', bool, False)
trn = 'data/train.list' if not is_predict else None
tst = 'data/test.list' if not is_predict else 'data/pred.list'
process = 'process' if not is_predict else 'process_predict'

# 1. read data
define_py_data_sources2(
    train_list=trn, test_list=tst, module='dataprovider', obj=process)

# 2. learning algorithm
batch_size = 5 if not is_predict else 1
settings(
    batch_size=batch_size,
    learning_rate=1e-3,
    learning_method=RMSPropOptimizer())

# 3. Network configuration
feature_num = 46
hid_num = 6
if not is_predict:
    x1 = data_layer(name='features1', size=feature_num)
    x2 = data_layer(name='features2', size=feature_num)
    y = data_layer(name='label', size=1)
    hidden1 = fc_layer(
        name='hidden', input=x1, size=hid_num, act=LinearActivation())
    hidden2 = LayerOutput('hidden', LayerType.FC_LAYER, x2, size=feature_num)
    y1 = fc_layer(name='output', input=hidden1, size=1, act=SigmoidActivation())
    y2 = LayerOutput('output', LayerType.FC_LAYER, hidden2, size=1)
    outputs(rank_cost(left=y1, right=y2, label=y))
else:
    x = data_layer(name='features', size=feature_num)
    hidden = fc_layer(
        name='hidden', input=x, size=hid_num, act=LinearActivation())
    y_pred = fc_layer(
        name='output', input=hidden, size=1, act=SigmoidActivation())
    outputs(y_pred)
