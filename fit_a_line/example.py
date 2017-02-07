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

import paddle.trainer_config_helpers as conf_helps
import paddle.v2.layers as layers
import paddle


def main():
    # get data_iter
    train_iter = paddle.dataset.Boston

    # network config
    x = layers.data_layer(name='x', size=13)
    y = layers.data_layer(name='y', size=1)
    y_predict = layers.fc_layer(
        input=x,
        param_attr=conf_helps.ParamAttr(name='w'),
        size=1,
        act=conf_helps.LinearActivation(),
        bias_attr=conf_helps.ParamAttr(name='b'))
    cost = layers.regression_cost(input=y_predict, label=y)

    # create optimizer
    optimizer = paddle.v2.optimizer.Adam(
        learning_rate=1e-4,
        batch_size=1000,
        model_average=conf_helps.ModelAverage(average_window=0.5),
        regularization=conf_helps.L2Regularization(rate=0.5))

    # create model
    model = paddle.model.create(network=cost)
    model.init_parameter(method="random")
    model.init_optimizer(optimizer=optimizer)

    evaluator = model.create_evaluator(type="acc")

    # train one epoch
    for batch in train_iter:
        model.forward(batch, is_predict=False)          # compute predictions
        model.evaluate(evaluator, batch.label)          # accumulate prediction accuracy
        evaluator.show_acc()                            # plot some figure about this model
        model.backward()                                # compute gradients
        model.update()                                  # update parameters using SGD


if __name__ == '__main__':
    main()
