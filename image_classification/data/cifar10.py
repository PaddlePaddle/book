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
            mean += batch['data'].sum(0)
            num += len(batch['data'])
        mean /= num
        print mean.size
        data = {"mean": mean, "size": mean.size}
        cPickle.dump(
            data, open("mean.meta", 'w'), protocol=cPickle.HIGHEST_PROTOCOL)


def create_data():
    train_set = [DATA + "/data_batch_%d" % (i + 1) for i in xrange(0, 5)]
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
