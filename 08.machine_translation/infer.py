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
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.layers as pd
from paddle.fluid.executor import Executor
import os

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


def decode(encoder_out):
    encoder_last = pd.sequence_last_step(input=encoder_out)
    encoder_last_projected = pd.fc(
        input=encoder_last, size=decoder_size, act='tanh')

    max_len = pd.fill_constant(shape=[1], dtype='int64', value=max_length)
    counter = pd.zeros(shape=[1], dtype='int64', force_cpu=True)

    init_ids = pd.data(name="init_ids", shape=[1], dtype="int64", lod_level=2)
    init_scores = pd.data(
        name="init_scores", shape=[1], dtype="float32", lod_level=2)

    # arrays to save selected ids and corresponding scores for each step
    ids_array = pd.create_array('int64')
    pd.array_write(init_ids, array=ids_array, i=counter)
    scores_array = pd.create_array('float32')
    pd.array_write(init_scores, array=scores_array, i=counter)

    # arrays to save states and context for each step
    state_array = pd.create_array('float32')
    pd.array_write(encoder_last_projected, array=state_array, i=counter)
    context_array = pd.create_array('float32')
    pd.array_write(encoder_last, array=state_array, i=counter)

    cond = pd.less_than(x=counter, y=max_len)
    while_op = pd.While(cond=cond)
    with while_op.block():
        pre_ids = pd.array_read(array=ids_array, i=counter)
        pre_score = pd.array_read(array=scores_array, i=counter)
        pre_state = pd.array_read(array=state_array, i=counter)
        pre_context = pd.array_read(array=context_array, i=counter)

        # cell calculations
        pre_ids_emb = pd.embedding(
            input=pre_ids,
            size=[target_dict_dim, word_dim],
            dtype='float32',
            is_sparse=is_sparse)
        decoder_inputs = pd.fc(
            input=[pre_ids_emb, pre_context],
            size=decoder_size * 3,
            bias_attr=False)
        current_state = pd.gru_unit(
            input=decoder_inputs, hidden=pre_state, size=decoder_size)
        current_state_with_lod = pd.lod_reset(x=current_state, y=pre_score)
        current_score = pd.fc(
            input=current_state, size=target_dict_dim, act='softmax')

        # beam search
        topk_scores, topk_indices = pd.topk(current_score, k=beam_size)
        accu_scores = pd.elementwise_add(
            x=pd.log(topk_scores), y=pd.reshape(pre_score, shape=[-1]), axis=0)
        selected_ids, selected_scores = pd.beam_search(
            pre_ids,
            pre_score,
            topk_indices,
            accu_scores,
            beam_size,
            end_id=10,
            level=0)

        pd.increment(x=counter, value=1, in_place=True)
        # update states
        pd.array_write(selected_ids, array=ids_array, i=counter)
        pd.array_write(selected_scores, array=scores_array, i=counter)
        # update rnn state by sequence_expand acting as gather
        current_state = pd.sequence_expand(current_state, selected_ids)
        current_context = pd.sequence_expand(pre_context, selected_ids)
        pd.array_write(current_state, array=state_array, i=counter)
        pd.array_write(current_context, array=context_array, i=counter)

        # update conditional variable 
        length_cond = pd.less_than(x=counter, y=array_len)
        finish_cond = pd.logical_not(pd.is_empty(x=selected_ids))
        pd.logical_and(x=length_cond, y=finish_cond, out=cond)

    translation_ids, translation_scores = pd.beam_search_decode(
        ids=ids_array, scores=scores_array, beam_size=beam_size, end_id=10)

    return translation_ids, translation_scores


def decode_main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = Executor(place)
    exe.run(framework.default_startup_program())

    encoder_out = encoder()
    translation_ids, translation_scores = decode(encoder_out)
    fluid.io.load_persistables(executor=exe, dirname=model_save_dir)

    init_ids_data = np.array([1 for _ in range(batch_size)], dtype='int64')
    init_scores_data = np.array(
        [1. for _ in range(batch_size)], dtype='float32')
    init_ids_data = init_ids_data.reshape((batch_size, 1))
    init_scores_data = init_scores_data.reshape((batch_size, 1))
    init_lod = [1] * batch_size
    init_lod = [init_lod, init_lod]

    init_ids = fluid.create_lod_tensor(init_ids_data, init_lod, place)
    init_scores = fluid.create_lod_tensor(init_scores_data, init_lod, place)

    test_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.test(dict_size), buf_size=1000),
        batch_size=batch_size)

    feed_order = ['src_word_id']
    feed_list = [
        framework.default_main_program().global_block().var(var_name)
        for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)

    for data in test_data():
        feed_data = map(lambda x: [x[0]], data)
        feed_dict = feeder.feed(feed_data)
        feed_dict['init_ids'] = init_ids
        feed_dict['init_scores'] = init_scores

        results = exe.run(
            framework.default_main_program(),
            feed=feed_dict,
            fetch_list=[translation_ids, translation_scores],
            return_numpy=False)

        result_ids = np.array(results[0])
        result_ids_lod = results[0].lod()
        result_scores = np.array(results[1])

        print("Original sentence:")
        print(" ".join([src_dict[w] for w in feed_data[0][0][1:-1]]))
        print("Translated score and sentence:")
        for i in xrange(beam_size):
            start_pos = result_ids_lod[1][i] + 1
            end_pos = result_ids_lod[1][i + 1]
            print("%d\t%.4f\t%s\n" % (
                i + 1, result_scores[end_pos - 1],
                " ".join([trg_dict[w] for w in result_ids[start_pos:end_pos]])))

        break


def main(use_cuda):
    decode_main(False)  # Beam Search does not support CUDA


if __name__ == '__main__':
    use_cuda = os.getenv('WITH_GPU', '0') != '0'
    main(use_cuda)
