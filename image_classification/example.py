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

import paddle

def main():
    paddle.init(...) # use_gpu and other command-line arguments

    batch_size=128

    # describe the VGG network
    input = paddle.layer.data(name="image")
    label = paddle.layer.data(name="label")
    conv1 = paddle.layer.conv(input, 64, 2, [0.3, 0], 3)
    conv2 = paddle.layer.conv(conv1, 128, 2, [0.4, 0])
    conv3 = paddle.layer.conv(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = paddle.layer.conv(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = paddle.layer.conv(conv4, 512, 3, [0.4, 0.4, 0])
    drop = paddle.layer.dropout(conv5, dropout_rate=0.5)
    fc1 = paddle.layer.fc(drop, size=512, act=LinearActivation())
    bn = paddle.layer.batch_norm(
        fc1, act=ReluActivation(), layer_attr=ExtraAttr(drop_rate=0.5))
    fc2 = paddle.layer.fc(bn, size=512, act=LinearActivation())
    out = paddle.layer.fc(fc2, size=10, act=SoftmaxActivation())

    # create model and loss function, optimizer
    model = paddle.model.create(out)
    cost = paddle.cost.classification(out, label)
    adam = paddle.optimizer.Adam(...) # learning_rate and other arguments
    adam.train(model, cost, ...) # some arguments for train, like trainer_count

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
