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

from paddle.trainer.PyDataProvider2 import *
import numpy as np
import struct


# Define a py data provider
@provider(
    input_types={'pixel': dense_vector(28 * 28),
                 'label': integer_value(10)})
def process(settings, filename):  # settings is not used currently.
    with open(filename + "-images-idx3-ubyte", "rb") as f:  # open picture file
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(
            f, 'ubyte',
            count=n * rows * cols).reshape(n, rows, cols).astype('float32')
        images = images / 255.0 * 2.0 - 1.0  # normalized to [-1,1]

    with open(filename + "-labels-idx1-ubyte", "rb") as l:  # open label file
        magic, n = struct.unpack(">II", l.read(8))
        labels = np.fromfile(l, 'ubyte', count=n).astype("int")

    for i in xrange(n):
        yield {"pixel": images[i, :], 'label': labels[i]}
