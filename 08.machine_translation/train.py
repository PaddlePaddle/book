import sys, os
import numpy as np
import paddle.v2 as paddle

with_gpu = os.getenv('WITH_GPU', '0') != '0'


def save_model(trainer, parameters, save_path):
    with open(save_path, 'w') as f:
        trainer.save_parameter_to_tar(f)


def seq_to_seq_net(source_dict_dim,
                   target_dict_dim,
                   is_generating,
                   beam_size=3,
                   max_length=250):
    ### Network Architecture
    word_vector_dim = 512  # dimension of word vector
    decoder_size = 512  # dimension of hidden unit of GRU decoder
    encoder_size = 512  # dimension of hidden unit of GRU encoder

    #### Encoder
    src_word_id = paddle.layer.data(
        name='source_language_word',
        type=paddle.data_type.integer_value_sequence(source_dict_dim))
    src_embedding = paddle.layer.embedding(
        input=src_word_id, size=word_vector_dim)
    src_forward = paddle.networks.simple_gru(
        input=src_embedding, size=encoder_size)
    src_backward = paddle.networks.simple_gru(
        input=src_embedding, size=encoder_size, reverse=True)
    encoded_vector = paddle.layer.concat(input=[src_forward, src_backward])

    #### Decoder
    encoded_proj = paddle.layer.fc(
        act=paddle.activation.Linear(),
        size=decoder_size,
        bias_attr=False,
        input=encoded_vector)

    backward_first = paddle.layer.first_seq(input=src_backward)

    decoder_boot = paddle.layer.fc(
        size=decoder_size,
        act=paddle.activation.Tanh(),
        bias_attr=False,
        input=backward_first)

    def gru_decoder_with_attention(enc_vec, enc_proj, current_word):

        decoder_mem = paddle.layer.memory(
            name='gru_decoder', size=decoder_size, boot_layer=decoder_boot)

        context = paddle.networks.simple_attention(
            encoded_sequence=enc_vec,
            encoded_proj=enc_proj,
            decoder_state=decoder_mem)

        decoder_inputs = paddle.layer.fc(
            act=paddle.activation.Linear(),
            size=decoder_size * 3,
            bias_attr=False,
            input=[context, current_word],
            layer_attr=paddle.attr.ExtraLayerAttribute(
                error_clipping_threshold=100.0))

        gru_step = paddle.layer.gru_step(
            name='gru_decoder',
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size)

        out = paddle.layer.fc(
            size=target_dict_dim,
            bias_attr=True,
            act=paddle.activation.Softmax(),
            input=gru_step)
        return out

    decoder_group_name = 'decoder_group'
    group_input1 = paddle.layer.StaticInput(input=encoded_vector)
    group_input2 = paddle.layer.StaticInput(input=encoded_proj)
    group_inputs = [group_input1, group_input2]

    if not is_generating:
        trg_embedding = paddle.layer.embedding(
            input=paddle.layer.data(
                name='target_language_word',
                type=paddle.data_type.integer_value_sequence(target_dict_dim)),
            size=word_vector_dim,
            param_attr=paddle.attr.ParamAttr(name='_target_language_embedding'))
        group_inputs.append(trg_embedding)

        # For decoder equipped with attention mechanism, in training,
        # target embeding (the groudtruth) is the data input,
        # while encoded source sequence is accessed to as an unbounded memory.
        # Here, the StaticInput defines a read-only memory
        # for the recurrent_group.
        decoder = paddle.layer.recurrent_group(
            name=decoder_group_name,
            step=gru_decoder_with_attention,
            input=group_inputs)

        lbl = paddle.layer.data(
            name='target_language_next_word',
            type=paddle.data_type.integer_value_sequence(target_dict_dim))
        cost = paddle.layer.classification_cost(input=decoder, label=lbl)

        return cost
    else:
        # In generation, the decoder predicts a next target word based on
        # the encoded source sequence and the previous generated target word.

        # The encoded source sequence (encoder's output) must be specified by
        # StaticInput, which is a read-only memory.
        # Embedding of the previous generated word is automatically retrieved
        # by GeneratedInputs initialized by a start mark <s>.

        trg_embedding = paddle.layer.GeneratedInput(
            size=target_dict_dim,
            embedding_name='_target_language_embedding',
            embedding_size=word_vector_dim)
        group_inputs.append(trg_embedding)

        beam_gen = paddle.layer.beam_search(
            name=decoder_group_name,
            step=gru_decoder_with_attention,
            input=group_inputs,
            bos_id=0,
            eos_id=1,
            beam_size=beam_size,
            max_length=max_length)

        return beam_gen


def main():
    paddle.init(use_gpu=with_gpu, trainer_count=1)
    is_generating = False

    # source and target dict dim.
    dict_size = 30000
    source_dict_dim = target_dict_dim = dict_size

    # train the network
    if not is_generating:
        # define optimize method and trainer
        optimizer = paddle.optimizer.Adam(
            learning_rate=5e-5,
            regularization=paddle.optimizer.L2Regularization(rate=8e-4))

        cost = seq_to_seq_net(source_dict_dim, target_dict_dim, is_generating)
        parameters = paddle.parameters.create(cost)

        trainer = paddle.trainer.SGD(
            cost=cost, parameters=parameters, update_equation=optimizer)
        # define data reader
        wmt14_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.wmt14.train(dict_size), buf_size=8192),
            batch_size=4)

        # define event_handler callback
        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 10 == 0:
                    print("\nPass %d, Batch %d, Cost %f, %s" %
                          (event.pass_id, event.batch_id, event.cost,
                           event.metrics))
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()

                if not event.batch_id % 10:
                    save_path = 'params_pass_%05d_batch_%05d.tar' % (
                        event.pass_id, event.batch_id)
                    save_model(trainer, parameters, save_path)

            if isinstance(event, paddle.event.EndPass):
                # save parameters
                save_path = 'params_pass_%05d.tar' % (event.pass_id)
                save_model(trainer, parameters, save_path)

        # start to train
        trainer.train(
            reader=wmt14_reader, event_handler=event_handler, num_passes=2)

    # generate a english sequence to french
    else:
        # use the first 3 samples for generation
        gen_data = []
        gen_num = 3
        for item in paddle.dataset.wmt14.gen(dict_size)():
            gen_data.append([item[0]])
            if len(gen_data) == gen_num:
                break

        beam_size = 3
        beam_gen = seq_to_seq_net(source_dict_dim, target_dict_dim,
                                  is_generating, beam_size)

        # get the trained model, whose bleu = 26.92
        parameters = paddle.dataset.wmt14.model()

        # prob is the prediction probabilities, and id is the prediction word.
        beam_result = paddle.infer(
            output_layer=beam_gen,
            parameters=parameters,
            input=gen_data,
            field=['prob', 'id'])

        # load the dictionary
        src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)

        gen_sen_idx = np.where(beam_result[1] == -1)[0]
        assert len(gen_sen_idx) == len(gen_data) * beam_size

        # -1 is the delimiter of generated sequences.
        # the first element of each generated sequence its length.
        start_pos, end_pos = 1, 0
        for i, sample in enumerate(gen_data):
            print(
                " ".join([src_dict[w] for w in sample[0][1:-1]])
            )  # skip the start and ending mark when printing the source sentence
            for j in xrange(beam_size):
                end_pos = gen_sen_idx[i * beam_size + j]
                print("%.4f\t%s" % (beam_result[0][i][j], " ".join(
                    trg_dict[w] for w in beam_result[1][start_pos:end_pos])))
                start_pos = end_pos + 2
            print("\n")


if __name__ == '__main__':
    main()
