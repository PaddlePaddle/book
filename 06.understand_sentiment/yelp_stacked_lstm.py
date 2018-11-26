# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import time
start = time.time()

import os
import paddle
import paddle.fluid as fluid
from functools import partial
import numpy as np

CLASS_DIM = 2
EMB_DIM = 128
HID_DIM = 512
STACKED_NUM = 3
BATCH_SIZE = 128
USE_GPU = True

TRAIN_FILE = '/paddle/daming_paddle_lab/book/06.understand_sentiment/yelp_review_100000.json'
# TEST_FILE =  '/paddle/daming_paddle_lab/book/06.understand_sentiment/10_reviews.json'
TEST_FILE =  '/paddle/daming_paddle_lab/book/06.understand_sentiment/yelp_review_4100000.json'


def stacked_lstm_net(data, input_dim, class_dim, emb_dim, hid_dim, stacked_num):
    assert stacked_num % 2 == 1

    emb = fluid.layers.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)

    fc1 = fluid.layers.fc(input=emb, size=hid_dim)
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)

    inputs = [fc1, lstm1]

    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hid_dim)
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc, size=hid_dim, is_reverse=(i % 2) == 0)
        inputs = [fc, lstm]

    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

    prediction = fluid.layers.fc(
        input=[fc_last, lstm_last], size=class_dim, act='softmax')
    return prediction


def inference_program(word_dict):
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

    dict_dim = len(word_dict)
    net = stacked_lstm_net(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM,
                           STACKED_NUM)
    return net


def train_program(word_dict):
    prediction = inference_program(word_dict)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return [avg_cost, accuracy]


def optimizer_func():
    return fluid.optimizer.Adagrad(learning_rate=0.002)


def train(use_cuda, train_program, params_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    print("Loading YELP word dict....")
    word_dict = paddle.dataset.yelp.word_dict(TRAIN_FILE)

    print("Reading YELP training data....")
    train_reader = paddle.batch(
        paddle.dataset.yelp.train(word_dict, TRAIN_FILE),batch_size=BATCH_SIZE)

    print("Reading YELP testing data....")
    test_reader = paddle.batch(
        paddle.dataset.yelp.test(word_dict, TEST_FILE), batch_size=BATCH_SIZE)


    trainer = fluid.Trainer(
        train_func=partial(train_program, word_dict),
        place=place,
        optimizer_func=optimizer_func)

    feed_order = ['words', 'label']

    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            if event.step % 10 == 0:
                avg_cost, acc = trainer.test(
                    reader=test_reader, feed_order=feed_order)

                print('Step {0}, Test Loss {1:0.2}, Acc {2:0.2}'.format(
                    event.step, avg_cost, acc))

                print("Step {0}, Epoch {1} Metrics {2}".format(
                    event.step, event.epoch, map(np.array, event.metrics)))

        elif isinstance(event, fluid.EndEpochEvent):
            trainer.save_params(params_dirname)

    trainer.train(
        num_epochs=1,
        event_handler=event_handler,
        reader=train_reader,
        feed_order=feed_order)


def infer(use_cuda, inference_program, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    word_dict = paddle.dataset.yelp.word_dict(TEST_FILE)

    inferencer = fluid.Inferencer(
        infer_func=partial(inference_program, word_dict),
        param_path=params_dirname,
        place=place)

    # Setup input by creating LoDTensor to represent sequence of words.
    # Here each word is the basic element of the LoDTensor and the shape of
    # each word (base_shape) should be [1] since it is simply an index to
    # look up for the corresponding word vector.
    # Suppose the length_based level of detail (lod) info is set to [[3, 4, 2]],
    # which has only one lod level. Then the created LoDTensor will have only
    # one higher level structure (sequence of words, or sentence) than the basic
    # element (word). Hence the LoDTensor will hold data for three sentences of
    # length 3, 4 and 2, respectively.
    # Note that lod info should be a list of lists.

    reviews_str = [
        "Love the staff, love the meat, love the place. Prepare for a long line around lunch or dinner hours. \n\nThey ask you how you want you meat, lean or something maybe, I can't remember. Just say you don't want it too fatty. \n\nGet a half sour pickle and a hot pepper. Hand cut french fries too.",  # 5
        "Super simple place but amazing nonetheless. It's been around since the 30's and they still serve the same thing they started with: a bologna and salami sandwich with mustard. \n\nStaff was very helpful and friendly.",  # 5
        "Small unassuming place that changes their menu every so often. Cool decor and vibe inside their 30 seat restaurant. Call for a reservation. \n\nWe had their beef tartar and pork belly to start and a salmon dish and lamb meal for mains. Everything was incredible! I could go on at length about how all the listed ingredients really make their dishes amazing but honestly you just need to go. \n\nA bit outside of downtown montreal but take the metro out and it's less than a 10 minute walk from the station.",  # 5
        "Lester's is located in a beautiful neighborhood and has been there since 1951. They are known for smoked meat which most deli's have but their brisket sandwich is what I come to montreal for. They've got about 12 seats outside to go along with the inside. \n\nThe smoked meat is up there in quality and taste with Schwartz's and you'll find less tourists at Lester's as well.",  # 5
        "Love coming here. Yes the place always needs the floor swept but when you give out  peanuts in the shell how won't it always be a bit dirty. \n\nThe food speaks for itself, so good. Burgers are made to order and the meat is put on the grill when you order your sandwich. Getting the small burger just means 1 patty, the regular is a 2 patty burger which is twice the deliciousness. \n\nGetting the Cajun fries adds a bit of spice to them and whatever size you order they always throw more fries (a lot more fries) into the bag.",  # 4
        "Had their chocolate almond croissant and it was amazing! So light and buttery and oh my how chocolaty.\n\nIf you're looking for a light breakfast then head out here. Perfect spot for a coffee\/latte before heading out to the old port",  # 4
        "Cycle Pub Las Vegas was a blast! Got a groupon and rented the bike for 11 of us for an afternoon tour. Each bar was more fun than the last. Downtown Las Vegas has changed so much and for the better. We had a wide age range in this group from early 20's to mid 50's and everyone had so much fun! Our driver Tony was knowledgable , friendly and just plain fun! Would recommend this to anyone looking to do something different away from the strip. You won't be disappointed!",  # 5
        "Who would have guess that you would be able to get fairly decent Vietnamese restaurant in East York? \n\nNot quite the same as Chinatown in terms of pricing (slightly higher) but definitely one of the better Vietnamese restaurants outside of the neighbourhood. When I don't have time to go to Chinatown, this is the next best thing as it is down the street from me.\n\nSo far the only items I have tried are the phos (beef, chicken & vegetarian) - and they have not disappointed me! Especially the chicken pho.\n\nNext time I go back, I'm going to try the banh cuon (steamed rice noodle) and the vermicelli!",  # 4
        "Terrible service and not so great drinks.\n\nWe happen to be in the plaza and saw this place for bubble tea, so we decided to grab a few to go.\n\nFirst of all, the two menus is located in a very awkward place (one very high up on the wall and one in the corner also by the wall), set up is not really friendly neither can it accommodate a large group of people to look at the menu at the same time.\n\nThe menu items itself had different options than your typically bubble tea place. Their specialty are mousse drinks - which we had no idea what exactly it was. So when we asked the staff, they said it was some sort of cheese. \n\nFor a Sunday afternoon, they were also out of grass jelly and pudding, which are ingredients in quite a few of their drinks (so not sure how that works).\n\nThe service is really bad, they don't understand English so when you asked them about the menu items, they can't really explain what they are.\n\nWe had ordered the following:\n- slush mango (came with tapioca and mousse)\n- peach ice tea\n- roasted oolong milk tea\n- supernova kumquat lemonade\n\nAnd when we asked what the difference between a supernova kumquat lemonade under the galaxy drinks section (which are supposed to be layered drinks), and the kumquat lemonade, they said it had a layer of blue mousse - so we were like ok we will try it then. But what came was just a bigger cup size of the kumquat lemonade. When we asked them again, they said it was just a size difference, that's what makes it supernova - which is pretty dumb to me as on the menu both items had the same two size choices of regular or large. \n\nThe mango slush and the roasted oolong milk tea wasn't bad. But the other two was not that great. The tea had a big floral taste that overpowered everything. There was no peach taste in the peach ice tea. The kumquat one had no kumquat taste and was just bitter from the lime and it tasted like it had no sugar in it at all.\n\nI guess if you need a non-busy place to use WIFI and work, then this is the place, no one wants to come here.\n\nAlso - their gimmick also seems to be the lightbulb drinks, which were intriguing to us, but you are not allowed to take the lightbulb home...\n\nWe will definitely not be returning...",  # 1
        "worse customer service ever. \nManager on duty was rude. She didn't care that I had negative feelings about this place when I said that I would never come back again!\nRestaurant has gone downhill since they renovated!!",  # 1
    ]
    # import pdb;pdb.set_trace()

    reviews = [c.split() for c in reviews_str]

    UNK = word_dict['<unk>']
    lod = []
    for c in reviews:
        lod.append([word_dict.get(words, UNK) for words in c])

    base_shape = [[len(c) for c in lod]]

    tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
    results = inferencer.infer({'words': tensor_words})

    for i, r in enumerate(results[0]):
        print("Predict probability of ", r[0], " to be positive and ", r[1],
              " to be negative for review \'", reviews_str[i], "\'")


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    params_dirname = "understand_sentiment_stacked_lstm.inference.model"
    train(use_cuda, train_program, params_dirname)
    infer(use_cuda, inference_program, params_dirname)

    finish = time.time()
    elapsed = finish - start
    print(elapsed)

if __name__ == '__main__':
    use_cuda = False # set to True if training with GPU
    main(use_cuda)
