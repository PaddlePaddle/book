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
import cPickle
from paddle.trainer.PyDataProvider2 import *


def initializer(settings, mean_path, is_train, **kwargs):
    settings.is_train = is_train
    settings.input_size = 3 * 32 * 32
    settings.mean = np.load(mean_path)['mean']
    settings.input_types = {
        'image': dense_vector(settings.input_size),
        'label': integer_value(10)
    }


@provider(init_hook=initializer, cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, file_list):
    with open(file_list, 'r') as fdata:
        for fname in fdata:
            fo = open(fname.strip(), 'rb')
            batch = cPickle.load(fo)
            fo.close()
            images = batch['data']
            labels = batch['labels']
            for im, lab in zip(images, labels):
                if settings.is_train and np.random.randint(2):
                    im = im[:, :, ::-1]
                im = im - settings.mean
                yield {'image': im.astype('float32'), 'label': int(lab)}
