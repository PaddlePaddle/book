_IDX = 2
START = "<s>"
END = "<e>"


def initialize_reader(settings, src_dict_path, trg_dict_path, is_generating,
                      file_list, **kwargs):
    def fun(dict_path):
        out_dict = dict()
        with open(dict_path, "r") as fin:
            out_dict = {
                line.strip(): line_count
                for line_count, line in enumerate(fin)
            }
        return out_dict

    settings.src_dict = fun(src_dict_path)
    settings.trg_dict = fun(trg_dict_path)

    settings.logger.info("src dict len : %d" % (len(settings.src_dict)))

    settings.slots = {
        'source_language_word': integer_value_sequence(len(settings.src_dict)),
        'target_language_word': integer_value_sequence(len(settings.trg_dict)),
        'target_language_next_word':
        integer_value_sequence(len(settings.trg_dict))
    }
    settings.logger.info("trg dict len : %d" % (len(settings.trg_dict)))


def _get_ids(s, dictionary):
    words = s.strip().split()
    return [dictionary[START]] + \
        [dictionary.get(w, UNK_IDX) for w in words] + \
        [dictionary[END]]


@provider(init_hook=initialize_reader, pool_size=50000)
def read(settings, file_name):
    with open(file_name, 'r') as f:
        for line_count, line in enumerate(f):
            line_split = line.strip().split('\t')
            if settings.job_mode and len(line_split) != 2:
                continue
            src_seq = line_split[0]  # one source sequence
            src_ids = _get_ids(src_seq, settings.src_dict)

            trg_seq = line_split[1]  # one target sequence
            trg_words = trg_seq.split()
            trg_ids = [settings.trg_dict.get(w, UNK_IDX) for w in trg_words]

            # remove sequence whose length > 80 in training mode
            if len(src_ids) > 80 or len(trg_ids) > 80:
                continue
            trg_ids_next = trg_ids + [settings.trg_dict[END]]
            trg_ids = [settings.trg_dict[START]] + trg_ids
            yield {
                'source_language_word': src_ids,
                'target_language_word': trg_ids,
                'target_language_next_word': trg_ids_next
            }


### Network Architecture
SOURCE_DICT_DIM = len(open(src_lang_dict, "r").readlines())
TARGET_DICT_DIM = len(open(trg_lang_dict, "r").readlines())
WORD_VECTOR_DIM = 512  # dimension of word vector
DECODER_SIZE = 512  # dimension of hidden unit in GRU Decoder network
ENCODER_SIZE = 512  # dimension of hidden unit in GRU Encoder network

#### Encoder
src_word_id = data_layer(name='source_language_word', size=SOURCE_DICT_DIM)
src_embedding = embedding_layer(
    input=src_word_id,
    size=WORD_VECTOR_DIM,
    param_attr=ParamAttr(name='_source_language_embedding'))
src_forward = simple_gru(input=src_embedding, size=ENCODER_SIZE)
src_backward = simple_gru(input=src_embedding, size=ENCODER_SIZE, reverse=True)
encoded_vector = concat_layer(input=[src_forward, src_backward])

#### Decoder
encoded_proj = mixed_layer(
    size=DECODER_SIZE, input=full_matrix_projection(input=encoded_vector))

backward_first = first_seq(input=src_backward)

decoder_boot = mixed_layer(
    size=DECODER_SIZE,
    act=TanhActivation(),
    input=full_matrix_projection(input=backward_first))


def gru_decoder_with_attention(enc_vec, enc_proj, current_word):
    decoder_mem = memory(
        name='gru_decoder', size=DECODER_SIZE, boot_layer=decoder_boot)

    context = simple_attention(
        encoded_sequence=enc_vec,
        encoded_proj=enc_proj,
        decoder_state=decoder_mem, )

    with mixed_layer(size=DECODER_SIZE * 3) as decoder_inputs:
        decoder_inputs += full_matrix_projection(input=context)
        decoder_inputs += full_matrix_projection(input=current_word)

    gru_step = gru_step_layer(
        name='gru_decoder',
        input=decoder_inputs,
        output_mem=decoder_mem,
        size=DECODER_SIZE)

    with mixed_layer(
            size=TARGET_DICT_DIM, bias_attr=True,
            act=SoftmaxActivation()) as out:
        out += full_matrix_projection(input=gru_step)
    return out


decoder_group_name = "decoder_group"
group_input1 = StaticInput(input=encoded_vector, is_seq=True)
group_input2 = StaticInput(input=encoded_proj, is_seq=True)
group_inputs = [group_input1, group_input2]

trg_embedding = embedding_layer(
    input=data_layer(
        name='target_language_word', size=TARGET_DICT_DIM),
    size=WORD_VECTOR_DIM,
    param_attr=ParamAttr(name='_target_language_embedding'))
group_inputs.append(trg_embedding)

# For decoder equipped with attention mechanism, in training,
# target embeding (the groudtruth) is the data input,
# while encoded source sequence is accessed to as an unbounded memory.
# Here, the StaticInput defines a read-only memory
# for the recurrent_group.
decoder = recurrent_group(
    name=decoder_group_name,
    step=gru_decoder_with_attention,
    input=group_inputs)

lbl = data_layer(name='target_language_next_word', size=TARGET_DICT_DIM)
cost = classification_cost(input=decoder, label=lbl)

train(cost, paddle.data.Reader(read, TRAINING_DATA_PATH), ipython_plot_widget)
