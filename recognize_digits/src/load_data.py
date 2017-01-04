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
import numpy as np
import matplotlib.pyplot as plt
import random


def read_data(path, filename):
    imgf = path + filename + "-images-idx3-ubyte"
    labelf = path + filename + "-labels-idx1-ubyte"
    f = open(imgf, "rb")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)

    # Define number of samples for train/test
    n = 60000 if "train" in filename else 10000

    rows = 28
    cols = 28

    images = np.fromfile(
        f, 'ubyte',
        count=n * rows * cols).reshape(n, rows, cols).astype('float32')
    labels = np.fromfile(l, 'ubyte', count=n).astype("int")

    return images, labels


if __name__ == "__main__":
    train_images, train_labels = read_data("./raw_data/", "train")
    test_images, test_labels = read_data("./raw_data/", "t10k")
    label_list = []
    for i in range(10):
        index = random.randint(0, train_images.shape[0] - 1)
        label_list.append(train_labels[index])
        plt.subplot(1, 10, i + 1)
        plt.imshow(train_images[index], cmap="Greys_r")
        plt.axis('off')
    print('label: %s' % (label_list, ))
    plt.show()
