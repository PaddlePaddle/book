import os
import numpy as np
import cPickle

DATA = "cifar-10-batches-py"
CHANNEL = 3
HEIGHT = 32
WIDTH = 32

def create_mean(dataset):
    if not os.path.isfile("mean.meta"):
        mean = np.zeros(CHANNEL * HEIGHT * WIDTH)
        num = 0
        for f in dataset:
            batch = np.load(f)
            mean +=  batch['data'].sum(0)
            num += len(batch['data'])
        mean /= num
        print mean.size
        data = {"mean": mean, "size": mean.size}
        cPickle.dump(data, open("mean.meta", 'w'), protocol=cPickle.HIGHEST_PROTOCOL)


def create_data():
    train_set = [DATA + "/data_batch_%d" % (i + 1) for i in xrange(0,5)]
    test_set = [DATA + "/test_batch"]

    # create mean values
    create_mean(train_set)

    # create dataset lists
    if not os.path.isfile("train.txt"):
        train = ["data/" + i for i in train_set]
        open("train.txt", "w").write("\n".join(train))
        open("train.list", "w").write("\n".join(["data/train.txt"]))

    if not os.path.isfile("text.txt"):
        test = ["data/" + i for i in test_set]
        open("test.txt", "w").write("\n".join(test))
        open("test.list", "w").write("\n".join(["data/test.txt"]))

if __name__ == '__main__':
    create_data()
