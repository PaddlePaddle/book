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
import os
import six

import numpy as np
import paddle
import paddle.fluid as fluid

dict_size = 30000
source_dict_size = target_dict_size = dict_size
word_dim = 512
hidden_dim = 512
decoder_size = hidden_dim
max_length = 256
beam_size = 4
batch_size = 64

is_sparse = True
model_save_dir = "machine_translation.inference.model"


def encoder():
    src_word_id = fluid.layers.data(
        name="src_word_id", shape=[1], dtype='int64', lod_level=1)
    src_embedding = fluid.layers.embedding(
        input=src_word_id,
        size=[source_dict_size, word_dim],
        dtype='float32',
        is_sparse=is_sparse)

    fc_forward = fluid.layers.fc(
        input=src_embedding, size=hidden_dim * 3, bias_attr=False)
    src_forward = fluid.layers.dynamic_gru(input=fc_forward, size=hidden_dim)
    fc_backward = fluid.layers.fc(
        input=src_embedding, size=hidden_dim * 3, bias_attr=False)
    src_backward = fluid.layers.dynamic_gru(
        input=fc_backward, size=hidden_dim, is_reverse=True)
    encoded_vector = fluid.layers.concat(
        input=[src_forward, src_backward], axis=1)
    return encoded_vector


def cell(x, hidden, encoder_out, encoder_out_proj):
    def simple_attention(encoder_vec, encoder_proj, decoder_state):
        decoder_state_proj = fluid.layers.fc(
            input=decoder_state, size=decoder_size, bias_attr=False)
        decoder_state_expand = fluid.layers.sequence_expand(
            x=decoder_state_proj, y=encoder_proj)
        mixed_state = fluid.layers.elementwise_add(encoder_proj,
                                                   decoder_state_expand)
        attention_weights = fluid.layers.fc(
            input=mixed_state, size=1, bias_attr=False)
        attention_weights = fluid.layers.sequence_softmax(
            input=attention_weights)
        weigths_reshape = fluid.layers.reshape(x=attention_weights, shape=[-1])
        scaled = fluid.layers.elementwise_mul(
            x=encoder_vec, y=weigths_reshape, axis=0)
        context = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
        return context

    context = simple_attention(encoder_out, encoder_out_proj, hidden)
    out = fluid.layers.fc(
        input=[x, context], size=decoder_size * 3, bias_attr=False)
    out = fluid.layers.gru_unit(
        input=out, hidden=hidden, size=decoder_size * 3)[0]
    return out, out


def train_decoder(encoder_out):
    encoder_last = fluid.layers.sequence_last_step(input=encoder_out)
    encoder_last_proj = fluid.layers.fc(
        input=encoder_last, size=decoder_size, act='tanh')
    # cache the encoder_out's computed result in attention
    encoder_out_proj = fluid.layers.fc(
        input=encoder_out, size=decoder_size, bias_attr=False)

    trg_language_word = fluid.layers.data(
        name="target_language_word", shape=[1], dtype='int64', lod_level=1)
    trg_embedding = fluid.layers.embedding(
        input=trg_language_word,
        size=[target_dict_size, word_dim],
        dtype='float32',
        is_sparse=is_sparse)

    rnn = fluid.layers.DynamicRNN()
    with rnn.block():
        x = rnn.step_input(trg_embedding)
        pre_state = rnn.memory(init=encoder_last_proj, need_reorder=True)
        encoder_out = rnn.static_input(encoder_out)
        encoder_out_proj = rnn.static_input(encoder_out_proj)
        out, current_state = cell(x, pre_state, encoder_out, encoder_out_proj)
        prob = fluid.layers.fc(input=out, size=target_dict_size, act='softmax')

        rnn.update_memory(pre_state, current_state)
        rnn.output(prob)

    return rnn()


def train_model():
    encoder_out = encoder()
    rnn_out = train_decoder(encoder_out)
    label = fluid.layers.data(
        name="target_language_next_word", shape=[1], dtype='int64', lod_level=1)
    cost = fluid.layers.cross_entropy(input=rnn_out, label=label)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def optimizer_func():
    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0))
    lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(hidden_dim, 1000)
    return fluid.optimizer.Adam(
        learning_rate=lr_decay,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=1e-4))


def train(use_cuda):
    train_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            avg_cost = train_model()
            optimizer = optimizer_func()
            optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt16.train(source_dict_size, target_dict_size),
            buf_size=10000),
        batch_size=batch_size)

    feeder = fluid.DataFeeder(
        feed_list=[
            'src_word_id', 'target_language_word', 'target_language_next_word'
        ],
        place=place,
        program=train_prog)

    exe.run(startup_prog)

    EPOCH_NUM = 20
    for pass_id in six.moves.xrange(EPOCH_NUM):
        batch_id = 0
        for data in train_data():
            cost = exe.run(
                train_prog, feed=feeder.feed(data), fetch_list=[avg_cost])[0]
            print('pass_id: %d, batch_id: %d, loss: %f' % (pass_id, batch_id,
                                                           cost))
            batch_id += 1
        fluid.io.save_params(exe, model_save_dir, main_program=train_prog)


def infer_decoder(encoder_out):
    encoder_last = fluid.layers.sequence_last_step(input=encoder_out)
    encoder_last_proj = fluid.layers.fc(
        input=encoder_last, size=decoder_size, act='tanh')
    encoder_out_proj = fluid.layers.fc(
        input=encoder_out, size=decoder_size, bias_attr=False)

    max_len = fluid.layers.fill_constant(
        shape=[1], dtype='int64', value=max_length)
    counter = fluid.layers.zeros(shape=[1], dtype='int64', force_cpu=True)

    init_ids = fluid.layers.data(
        name="init_ids", shape=[1], dtype="int64", lod_level=2)
    init_scores = fluid.layers.data(
        name="init_scores", shape=[1], dtype="float32", lod_level=2)
    # create and init arrays to save selected ids, scores and states for each step
    ids_array = fluid.layers.array_write(init_ids, i=counter)
    scores_array = fluid.layers.array_write(init_scores, i=counter)
    state_array = fluid.layers.array_write(encoder_last_proj, i=counter)

    cond = fluid.layers.less_than(x=counter, y=max_len)
    while_op = fluid.layers.While(cond=cond)
    with while_op.block():
        pre_ids = fluid.layers.array_read(array=ids_array, i=counter)
        pre_score = fluid.layers.array_read(array=scores_array, i=counter)
        pre_state = fluid.layers.array_read(array=state_array, i=counter)

        pre_ids_emb = fluid.layers.embedding(
            input=pre_ids,
            size=[target_dict_size, word_dim],
            dtype='float32',
            is_sparse=is_sparse)
        out, current_state = cell(pre_ids_emb, pre_state, encoder_out,
                                  encoder_out_proj)
        prob = fluid.layers.fc(
            input=current_state, size=target_dict_size, act='softmax')

        # beam search
        topk_scores, topk_indices = fluid.layers.topk(prob, k=beam_size)
        accu_scores = fluid.layers.elementwise_add(
            x=fluid.layers.log(topk_scores),
            y=fluid.layers.reshape(pre_score, shape=[-1]),
            axis=0)
        accu_scores = fluid.layers.lod_reset(x=accu_scores, y=pre_ids)
        selected_ids, selected_scores = fluid.layers.beam_search(
            pre_ids, pre_score, topk_indices, accu_scores, beam_size, end_id=1)

        fluid.layers.increment(x=counter, value=1, in_place=True)
        # save selected ids and corresponding scores of each step
        fluid.layers.array_write(selected_ids, array=ids_array, i=counter)
        fluid.layers.array_write(selected_scores, array=scores_array, i=counter)
        # update rnn state by sequence_expand acting as gather
        current_state = fluid.layers.sequence_expand(current_state,
                                                     selected_ids)
        fluid.layers.array_write(current_state, array=state_array, i=counter)
        current_enc_out = fluid.layers.sequence_expand(encoder_out,
                                                       selected_ids)
        fluid.layers.assign(current_enc_out, encoder_out)
        current_enc_out_proj = fluid.layers.sequence_expand(encoder_out_proj,
                                                            selected_ids)
        fluid.layers.assign(current_enc_out_proj, encoder_out_proj)

        # update conditional variable
        length_cond = fluid.layers.less_than(x=counter, y=max_len)
        finish_cond = fluid.layers.logical_not(
            fluid.layers.is_empty(x=selected_ids))
        fluid.layers.logical_and(x=length_cond, y=finish_cond, out=cond)

    translation_ids, translation_scores = fluid.layers.beam_search_decode(
        ids=ids_array, scores=scores_array, beam_size=beam_size, end_id=1)

    return translation_ids, translation_scores


def infer_model():
    encoder_out = encoder()
    translation_ids, translation_scores = infer_decoder(encoder_out)
    return translation_ids, translation_scores


def infer(use_cuda):
    infer_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            translation_ids, translation_scores = infer_model()

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    test_data = paddle.batch(
        paddle.dataset.wmt16.test(source_dict_size, target_dict_size),
        batch_size=batch_size)
    src_idx2word = paddle.dataset.wmt16.get_dict(
        "en", source_dict_size, reverse=True)
    trg_idx2word = paddle.dataset.wmt16.get_dict(
        "de", target_dict_size, reverse=True)

    fluid.io.load_params(exe, model_save_dir, main_program=infer_prog)

    for data in test_data():
        src_word_id = fluid.create_lod_tensor(
            data=[x[0] for x in data],
            recursive_seq_lens=[[len(x[0]) for x in data]],
            place=place)
        init_ids = fluid.create_lod_tensor(
            data=np.array([[0]] * len(data), dtype='int64'),
            recursive_seq_lens=[[1] * len(data)] * 2,
            place=place)
        init_scores = fluid.create_lod_tensor(
            data=np.array([[0.]] * len(data), dtype='float32'),
            recursive_seq_lens=[[1] * len(data)] * 2,
            place=place)
        seq_ids, seq_scores = exe.run(
            infer_prog,
            feed={
                'src_word_id': src_word_id,
                'init_ids': init_ids,
                'init_scores': init_scores
            },
            fetch_list=[translation_ids, translation_scores],
            return_numpy=False)
        # How to parse the results:
        #   Suppose the lod of seq_ids is:
        #     [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
        #   then from lod[0]:
        #     there are 2 source sentences, beam width is 3.
        #   from lod[1]:
        #     the first source sentence has 3 hyps; the lengths are 12, 12, 16
        #     the second source sentence has 3 hyps; the lengths are 14, 13, 15
        hyps = [[] for i in range(len(seq_ids.lod()[0]) - 1)]
        scores = [[] for i in range(len(seq_scores.lod()[0]) - 1)]
        for i in range(len(seq_ids.lod()[0]) - 1):  # for each source sentence
            start = seq_ids.lod()[0][i]
            end = seq_ids.lod()[0][i + 1]
            print("Original sentence:")
            print(" ".join([src_idx2word[idx] for idx in data[i][0][1:-1]]))
            print("Translated score and sentence:")
            for j in range(end - start):  # for each candidate
                sub_start = seq_ids.lod()[1][start + j]
                sub_end = seq_ids.lod()[1][start + j + 1]
                hyps[i].append(" ".join([
                    trg_idx2word[idx]
                    for idx in np.array(seq_ids)[sub_start:sub_end][1:-1]
                ]))
                scores[i].append(np.array(seq_scores)[sub_end - 1])
                print(scores[i][-1], hyps[i][-1].encode('utf8'))


def main(use_cuda):
    train(use_cuda)
    infer(use_cuda)


if __name__ == '__main__':
    use_cuda = False  # set to True if training with GPU
    main(use_cuda)
