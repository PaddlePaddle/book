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


#Define a data provider for "query relationship"
@provider(
    input_types={
        'features1': dense_vector(46),
        'features2': dense_vector(46),
        'label': dense_vector(1)
    },
    should_shuffle=False,
    cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, file_name):
    with open(file_name) as f:
        pre_qid = -1
        feats1 = []
        feats2 = []
        l1 = 0
        l2 = 0
        for line in f:
            line = line.split('#')[0]
            if len(line.split()) < 48:
                continue
            qid = int(line.split()[1].split(':')[1])
            if pre_qid != qid:
                feats1 = []
                for term in line.split()[2:48]:
                    feats1.append(float(term.split(':')[1]))
                l1 = int(line.split()[0])
                pre_qid = qid
                feats2 = feats1
                yield feats1, feats2, [0.5]
            else:
                feats1 = feats2
                feats2 = []
                l1 = l2
                for term in line.split()[2:48]:
                    feats2.append(float(term.split(':')[1]))
                l2 = int(line.split()[0])
                p12 = 0.5
                if l1 > l2:
                    p12 = 1
                if l1 < l2:
                    p12 = 0
                yield feats1, feats2, [p12]


@provider(input_types={'features': dense_vector(46)})
def process_predict(settings, file_name):
    with open(file_name) as f:
        for line in f:
            feats = []
            line = line.split('#')[0]
            for term in line.split()[2:48]:
                feats.append(float(term.split(':')[1]))
            yield feats
