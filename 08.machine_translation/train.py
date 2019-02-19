#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as pd
import os
import sys
try:
    from paddle.fluid.contrib.trainer import *
    from paddle.fluid.contrib.inferencer import *
except ImportError:
    print(
        "In the fluid 1.0, the trainer and inferencer are moving to paddle.fluid.contrib",
        file=sys.stderr)
    from paddle.fluid.trainer import *
    from paddle.fluid.inferencer import *

dict_size = 30000
source_dict_dim = target_dict_dim = dict_size
word_dim = 32
hidden_dim = 32
decoder_size = hidden_dim
max_length = 8
beam_size = 2
batch_size = 2

is_sparse = True
model_save_dir = "machine_translation.inference.model"


def encoder():
    src_word_id = pd.data(
        name="src_word_id", shape=[1], dtype='int64', lod_level=1)
    src_embedding = pd.embedding(
        input=src_word_id,
        size=[source_dict_dim, word_dim],
        dtype='float32',
        is_sparse=is_sparse)

    fc_forward = pd.fc(
        input=src_embedding, size=hidden_dim * 3, bias_attr=False)
    src_forward = pd.dynamic_gru(input=fc_forward, size=hidden_dim)
    fc_backward = pd.fc(
        input=src_embedding, size=hidden_dim * 3, bias_attr=False)
    src_backward = pd.dynamic_gru(
        input=fc_backward, size=hidden_dim, is_reverse=True)
    encoded_vector = pd.concat(input=[src_forward, src_backward], axis=1)
    return encoded_vector


def train_decoder(encoder_out):
    encoder_last = pd.sequence_last_step(input=encoder_out)
    encoder_last_projected = pd.fc(
        input=encoder_last, size=decoder_size, act='tanh')
    trg_language_word = pd.data(
        name="trg_word_id", shape=[1], dtype='int64', lod_level=1)
    trg_embedding = pd.embedding(
        input=trg_language_word,
        size=[target_dict_dim, word_dim],
        dtype='float32',
        is_sparse=is_sparse)

    rnn = pd.DynamicRNN()
    with rnn.block():
        current_word = rnn.step_input(trg_embedding)
        context = rnn.static_input(encoder_last)
        pre_state = rnn.memory(
            init=encoder_last_projected, size=decoder_size, need_reorder=True)

        decoder_inputs = pd.fc(
            input=[current_word, context],
            size=decoder_size * 3,
            bias_attr=False)
        current_state = pd.gru_unit(
            input=decoder_inputs, hidden=pre_state, size=decoder_size)

        current_score = pd.fc(
            input=current_state, size=target_dict_dim, act='softmax')

        rnn.update_memory(pre_state, current_state)
        rnn.output(current_score)

    return rnn()


def train_program():
    encoder_out = encoder()
    rnn_out = train_decoder(encoder_out)
    label = pd.data(
        name="trg_next_word_id", shape=[1], dtype='int64', lod_level=1)
    cost = pd.cross_entropy(input=rnn_out, label=label)
    avg_cost = pd.mean(cost)
    return avg_cost


def optimizer_func():
    return fluid.optimizer.Adagrad(
        learning_rate=1e-4,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.1))


def train(use_cuda):
    EPOCH_NUM = 1

    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)

    feed_order = [
        'src_word_id', 'target_language_word', 'target_language_next_word'
    ]

    def event_handler(event):
        if isinstance(event, EndStepEvent):
            if event.step % 10 == 0:
                print('pass_id=' + str(event.epoch) + ' batch=' + str(
                    event.step))

        if isinstance(event, EndEpochEvent):
            trainer.save_params(model_save_dir)

    trainer = Trainer(
        train_func=train_program, place=place, optimizer_func=optimizer_func)

    trainer.train(
        reader=train_reader,
        num_epochs=EPOCH_NUM,
        event_handler=event_handler,
        feed_order=feed_order)


def main(use_cuda):
    train(use_cuda)


if __name__ == '__main__':
    use_cuda = False  # set to True if training with GPU
    main(use_cuda)
