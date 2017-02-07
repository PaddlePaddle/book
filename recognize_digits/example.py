"""
A very basic example for how to use current Raw SWIG API to train mnist network.

Current implementation uses Raw SWIG, which means the API call is directly \
passed to C++ side of Paddle.

The user api could be simpler and carefully designed.
"""

import paddle.v2 as paddle

from mnist_util import read_from_mnist


def main():
    paddle.raw.initPaddle("-use_gpu=false",
                          "-trainer_count=4")  # use 4 cpu cores

    optimizer = paddle.optimizer.Optimizer(
        learning_method=paddle.optimizer.AdamOptimizer(),
        learning_rate=1e-4,
        model_average=paddle.optimizer.ModelAverage(average_window=0.5),
        regularization=paddle.optimizer.L2Regularization(rate=0.5),
        batch_size=128)  # 原来settings包含了batch_size

    # define network
    imgs = paddle.layers.data_layer(name='pixel', size=784)
    hidden1 = paddle.layers.fc_layer(input=imgs, size=200)
    hidden2 = paddle.layers.fc_layer(input=hidden1, size=200)
    inference = paddle.layers.fc_layer(
        input=hidden2, size=10, act=paddle.config.SoftmaxActivation())
    cost = paddle.layers.classification_cost(
        input=inference, label=paddle.layers.data_layer(
            name='label', size=10))

    model = paddle.model.Model(layers=[cost], optimizer=optimizer)

    model.rand_parameter()

    # 希望把train_data和test_data写在一块，因为两者的处理逻辑很相似
    # 去掉原来api设计中的model变量，这部分和model不相关
    train_data, test_data = paddle.data.create_data_pool(
        file_reader=read_from_mnist,
        file_list=['./data/raw_data/train', './data/raw_data/test'],
        shuffle=True)

    # Training process.
    model.start()

    # 去掉多余的start和finish函数，也去掉原来的make_evaluator
    # 能否把返回值直接作为evaluator输出呢？
    for pass_id in xrange(2):
        for batch_id, data_batch in enumerate(train_data):
            batch_evaluator = model.train(data_batch)
            print "Pass=%d, batch=%d" % (pass_id, batch_id), batch_evaluator

        for _, data_batch in enumerate(test_data):
            test_evaluator = model.test(data_batch)
        print "TEST Pass=%d" % pass_id, test_evaluator

        # 将训练和测试的cost画在一张图上
        plot_cost(batch_evaluator.cost, test_evaluator.cost, pass_id)

    model.finish()


if __name__ == '__main__':
    main()
