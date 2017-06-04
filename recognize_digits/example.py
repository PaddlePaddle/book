"""
A very basic example for how to use current Raw SWIG API to train mnist network.

Current implementation uses Raw SWIG, which means the API call is directly \
passed to C++ side of Paddle.

The user api could be simpler and carefully designed.
"""

import paddle

from mnist_util import read_mnist_data


def main():
    paddle.init("-use_gpu=false", "-trainer_count=4")  # use 4 cpu cores

    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-4,
        model_average=paddle.modelAverage(average_window=0.5),
        regularization=paddle.regularization.L2(rate=0.5),
        batch_size=128)

    # define network
    imgs = paddle.layer.data(name='pixel', size=784)
    hidden1 = paddle.layer.fc(input=imgs, size=200)
    hidden2 = paddle.layer.fc(input=hidden1, size=200)
    inference = paddle.layer.fc(input=hidden2,
                                size=10,
                                act=paddle.activation.Softmax())
    cost = paddle.cost.classification(
        input=inference, label=paddle.layer.data(
            name='label', size=10))

    model = paddle.model.create(network=inference)

    model.rand_parameter()

    train_data, test_data = paddle.data.create_data_pool(
        file_reader=read_from_mnist,
        file_list=['./data/raw_data/train', './data/raw_data/test'],
        shuffle=True)

    optimizer.train(model=model, cost=cost, data=[train_data, test_data])


if __name__ == '__main__':
    main()
