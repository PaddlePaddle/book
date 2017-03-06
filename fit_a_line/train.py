import paddle.v2 as paddle
import paddle.v2.dataset.uci_housing as uci_housing


def main():
    # init
    paddle.init(use_gpu=False, trainer_count=1)

    # network config
    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(13))
    y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())
    y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))
    cost = paddle.layer.regression_cost(input=y_predict, label=y)

    # create parameters
    parameters = paddle.parameters.create(cost)

    # create optimizer
    optimizer = paddle.optimizer.Momentum(momentum=0)

    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

    feeding = {'x': 0, 'y': 1}

    # event_handler to print training and testing info
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)

        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(
                reader=paddle.batch(
                    uci_housing.test(), batch_size=2),
                feeding=feeding)
            print "Test %d, Cost %f" % (event.pass_id, result.cost)

    # training
    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(
                uci_housing.train(), buf_size=500),
            batch_size=2),
        feeding=feeding,
        event_handler=event_handler,
        num_passes=30)


if __name__ == '__main__':
    main()
