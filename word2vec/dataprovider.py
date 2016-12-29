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
import collections
import logging
import pdb

logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s', )
logger = logging.getLogger('paddle')
logger.setLevel(logging.INFO)

N = 5  # Ngram
cutoff = 50  # select words with frequency > cutoff to dictionary


def build_dict(ftrain, fdict):
    sentences = []
    with open(ftrain) as fin:
        for line in fin:
            line = ['<s>'] + line.strip().split() + ['<e>']
            sentences += line
    wordfreq = collections.Counter(sentences)
    wordfreq = filter(lambda x: x[1] > cutoff, wordfreq.items())
    dictionary = sorted(wordfreq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*dictionary))
    for word in words:
        print >> fdict, word
    word_idx = dict(zip(words, xrange(len(words))))
    logger.info("Dictionary size=%s" % len(words))
    return word_idx


def initializer(settings, srcText, dictfile, **xargs):
    with open(dictfile, 'w') as fdict:
        settings.dicts = build_dict(srcText, fdict)
    input_types = []
    for i in xrange(N):
        input_types.append(integer_value(len(settings.dicts)))
    settings.input_types = input_types


@provider(init_hook=initializer)
def process(settings, filename):
    UNKID = settings.dicts['<unk>']
    with open(filename) as fin:
        for line in fin:
            line = ['<s>'] * (N - 1) + line.strip().split() + ['<e>']
            line = [settings.dicts.get(w, UNKID) for w in line]
            for i in range(N, len(line) + 1):
                yield line[i - N:i]
