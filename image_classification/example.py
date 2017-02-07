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


def main():
    paddle.init(...) # use_gpu and other command-line arguments

    batch_size=128

    # describe the VGG network
    input = paddle.data_layer(name="image")
    label = paddle.data_layer(name="label")
    conv1 = paddle.conv_layer(input, 64, 2, [0.3, 0], 3)
    conv2 = paddle.conv_layer(conv1, 128, 2, [0.4, 0])
    conv3 = paddle.conv_layer(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = paddle.conv_layer(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = paddle.conv_layer(conv4, 512, 3, [0.4, 0.4, 0])
    drop = paddle.dropout_layer(conv5, dropout_rate=0.5)
    fc1 = paddle.fc_layer(drop, size=512, act=LinearActivation())
    bn = paddle.batch_norm_layer(
        fc1, act=ReluActivation(), layer_attr=ExtraAttr(drop_rate=0.5))
    fc2 = paddle.fc_layer(bn, size=512, act=LinearActivation())
    out = paddle.fc_layer(fc2, size=10, act=SoftmaxActivation())
    cost = paddle.classification_cost(out, label)

    # optimizer
    optimizer = paddle.optimizer.Optimizer(
        learning_rate=0.1 / batch_size,
        learning_rate_decay_a=0.1,
        learning_rate_decay_b=50000 * 100,
        learning_rate_schedule='discexp',
        learning_method=MomentumOptimizer(0.9),
        regularization=L2Regularization(0.0005 * 128))

    # create paddle model
    model = paddle.model(...) # some arguments for model, like trainer_count
    model.network.init(cost)
    model.optimizer.init(optimizer)
    model.parameter.init()

    # get data from...
    train_data = paddle.data.create_data_pool(
        file_reader,
        file_list,
        model=model,
        batch_size=batch_size)
    test_data = paddle.data.create_data_pool(
        file_reader,
        file_list,
        model=model,
        batch_size=batch_size)

    # calculate
    for i in xrange(10):
      for batch_id, data_batch in enumerate(train_data):
        # backward表示训练模型参数
        model.calculate(data_batch, backward)

        # forward,这里表示计算到output layer为止
        model.calculate(data_batch, forward, "output"))

      model.parameter.save(...)


if __name__ == '__main__':
    main()
