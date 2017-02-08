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

import paddle.dataset
import paddle.layer
import paddle.optimizer
import paddle.model
import paddle.cost
import paddle.regularization


def main():
    # get data
    train_data = paddle.dataset.Boston

    # network config
    x = paddle.layer.data(name='x', size=13)
    y = paddle.layer.data(name='y', size=1)
    y_predict = paddle.layer.fc(
        input=x,
        param_attr=paddle.parameter.Attr(name='w'),
        size=1,
        act=paddle.activation.Linear(),
        bias_attr=paddle.parameter.Attr(name='b'))

    # define cost
    cost = paddle.cost.regression(input=y_predict, label=y)

    # create optimizer
    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-4,
        batch_size=1000,
        model_average=paddle.layer.ModelAverage(average_window=0.5),
        regularization=paddle.regularization.L2(rate=0.5))

    # create model
    model = paddle.model.create(network=y_predict)

    # train model
    optimizer.train(model=model, cost=cost, data=train_data)


if __name__ == '__main__':
    main()
