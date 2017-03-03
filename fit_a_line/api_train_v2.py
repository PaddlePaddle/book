import paddle.v2 as paddle
import paddle.v2.dataset.uci_housing as uci_housing
import cPickle
from paddle.proto.ParameterConfig_pb2 import ParameterConfig

def main():
    # init
    paddle.init(use_gpu=False, trainer_count=1)

    # network config
    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(13))
    y_predict = paddle.layer.fc(input=x,
                                param_attr=paddle.attr.Param(name='w'),
                                size=1,
                                act=paddle.activation.Linear(),
                                bias_attr=paddle.attr.Param(name='b'))
    y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))
    cost = paddle.layer.regression_cost(input=y_predict, label=y)

    # create parameters
    parameters = paddle.parameters.create(cost)

    # create optimizer
    optimizer = paddle.optimizer.Momentum(momentum=0)

    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

    reader_dict={'x': 0,
                 'y': 1}

    # event_handler to print training and testing info
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)

        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(
                reader=paddle.reader.batched(
                    uci_housing.test(), batch_size=2),
                reader_dict=reader_dict)
            print "Test %d, Cost %f" % (event.pass_id, result.cost)

    # training
    trainer.train(
        reader=paddle.reader.batched(
            paddle.reader.shuffle(
                uci_housing.train(), buf_size=500),
            batch_size=2),
        reader_dict=reader_dict,
        event_handler=event_handler,
        num_passes=30)

    #infer
    probs = paddle.infer(
        output=y_predict,
        parameters=parameters,
        reader=paddle.reader.batched(
                uci_housing.test(), batch_size=2),
                reader_dict=reader_dict)
    print probs.shape

if __name__ == '__main__':
    main()
