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
"""
Yelp dataset.

This module downloads IMDB dataset from
http://ai.stanford.edu/%7Eamaas/data/sentiment/. This dataset contains a set
of 25,000 highly polar movie reviews for training, and 25,000 for testing.
Besides, this module also provides API for building dictionary.
"""

import collections
import string
import re
import json
import unicodedata


def lazy_read(filename):
    with open(filename) as fp:
        while True:
            line = fp.readline()
            if not line:
                fp.close()
                break

            parsed_json = json.loads(line)
            if 'text' not in parsed_json:
                continue

            # we do not learn neutral ratings, only neg (1,2) and pos (4,5)
            if 'stars' not in parsed_json or parsed_json['stars'] == 3:
                continue

            label = 0
            if parsed_json['stars'] > 3:
                label = 1

            text = parsed_json['text']
            text = unicodedata.normalize('NFKD', text).encode('ascii','ignore')
            tokenized = text.rstrip("\n\r").translate(None, string.punctuation).lower().split()

            yield (tokenized, label)


def build_dict(filename, cutoff=1):
    """
    Build a word dictionary from the corpus. Keys of the dictionary are words,
    and values are zero-based IDs of these words.
    """
    word_freq = collections.defaultdict(int)
    for doc in lazy_read(filename):
        for word in doc[0]:
            word_freq[word] += 1

    # Not sure if we should prune less-frequent words here.
    word_freq = filter(lambda x: x[1] > cutoff, word_freq.items())

    dictionary = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*dictionary))
    word_idx = dict(zip(words, xrange(len(words))))
    word_idx['<unk>'] = len(words)
    return word_idx


def word_dict(filename):
    """
    Build a word dictionary from the corpus.

    :return: Word dictionary
    :rtype: dict
    """
    built_word_dict = build_dict(filename)
    # import pdb;pdb.set_trace()
    return built_word_dict


def reader_creator(word_idx, filename):
    UNK = word_idx['<unk>']
    INS = []

    def load(filename, out):
        for doc in lazy_read(filename):
            out.append(([word_idx.get(w, UNK) for w in doc[0]], doc[1]))

    load(filename, INS)

    def reader():
        for doc, label in INS:
            yield doc, label

    return reader


def train(word_idx, filename):
    """
    IMDB training set creator.

    It returns a reader creator, each sample in the reader is an zero-based ID
    sequence and label in [0, 1].

    :param word_idx: word dictionary
    :type word_idx: dict
    :return: Training reader creator
    :rtype: callable
    """
    return reader_creator(word_idx, filename)


def test(word_idx, filename):
    """
    IMDB test set creator.

    It returns a reader creator, each sample in the reader is an zero-based ID
    sequence and label in [0, 1].

    :param word_idx: word dictionary
    :type word_idx: dict
    :return: Test reader creator
    :rtype: callable
    """
    return reader_creator(word_idx, filename)


