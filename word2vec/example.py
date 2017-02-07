import paddle
import sklearn.manifold.TSNE


def plot(weight, words, canvas):
    """
    Visualize word2vec result.
    :param weight: the embedding matrix. Height is the word count, width is the
                   embedding dimension.
    :type weight: numpy.ndarray
    :param words: words with id and string.
    :param canvas:
    :return: Nothing
    """

    points = [w[1] for w in words]

    model = TSNE(n_components=2)
    points = model.fit_transform(points)  # in 2 dims, embedding result.

    canvas.plot(points, label=[w[0] for w in words])


def plot_train_error(*args):
    pass


def plot_test_error(*args):
    pass


def main():
    # Fetch Penn Tree Bank Dataset.
    # This api will download PTB dataset to a default place, and
    # read the dataset into a Python object.
    #
    # dataset.word_count is a list of a tuple. First value of this
    # tuple is a word, second value is the word count.
    # such as
    #   dataset.word_count[0] = ('a', 39450), dataset.word_count[1] = ('is', 34950)
    #   dataset.word_count is sorted by word_count desc.
    #
    # dataset.train_data(tokenized=True) will return all sentences in train set.
    #   if tokenized == True:
    #      return all sentences, each sentence is a list of word string.
    #   else:
    #      return all sentences, each sentence is a string.
    dataset = paddle.data.PTB.load()

    # Not all words in dataset will be learning into a embedding, because that
    # will be infinite words in a language. We only learning `word_dict_size`
    # embeddings. Other words, we mark them as UNK, which means `unknown key`.
    START = 0
    END = 1
    UNK = 2
    word_dict_size = 1950
    word_dict = {
        '<s>': START,
        '<e>': END
    }

    # Construct Dictionary.
    for i, each in dataset.word_count:
        if i == word_dict_size:
            break
        word_dict[each[0]] = i + UNK + 1  # word id start with `UNK+1`, 3.

    words = [None] * 5  # reserve a list of 5 elements.

    for i in xrange(len(words)):
        # Network has 5 inputs.
        # which name are word_0, word_1, word_2, word_3, word_4.
        # Size is word_dict length + UNK
        words[i] = paddle.layers.data(name='word_%d' % i,
                                      size=len(word_dict) + 1)

    # The Embedding file name.
    embedding_param_attr = paddle.layers.ParamAttr(name="embedding")
    embedding_size = 64

    # Ngram Model use w0,w1,w3,w4 to predict w5
    contexts = []
    for i in xrange(int(len(words) / 2)):
        contexts.append(
            paddle.layers.table_projection(input=words[i], size=embedding_size,
                                           param_attr=embedding_param_attr))
        contexts.append(
            paddle.layers.table_projection(input=words[-i], size=embedding_size,
                                           param_attr=embedding_param_attr)
        )

    # concat w0, w1, w3, w4 to a whole vector.
    context_embedding = paddle.layers.concat(input=contexts)
    hidden = paddle.layers.fc_layer(input=context_embedding, size=256,
                                    act=paddle.acts.SigmoidActivation())
    prediction = paddle.layers.fc_layer(input=hidden, size=len(word_dict) + 1,
                                        act=paddle.acts.SoftmaxActivation())

    cost = paddle.layers.classification_cost(input=prediction, label=words[
        int(len(words) / 2) + 1])

    model = paddle.model.create_model(cost)

    model.rand_parameter()  # init random.

    params = model.get_parameter(name='embedding')  # get embedding.

    # when training, the following words will be watched, ploted.
    watched_words = ['king', 'queen', 'man', 'woman']
    plot(params.get_weight(),
         [(word_dict.get(w, UNK), w) for w in watched_words],
         canvas=None)

    train_data = dataset.train_data(tokenized=True)
    test_data = dataset.test_data(tokenized=True)

    for each_epoch_id in xrange(10):  # train 10 epochs
        batch_size = 256
        batch_count = int(len(train_data) / 256) + 1
        for each_batch_id in xrange(batch_count):
            data_batch = train_data[each_batch_id * batch_size: min(
                (each_batch_id + 1) * batch_size, len(train_data))]

            loss = model.train(data_batch)
            if each_batch_id % 100 == 0:
                # get the classification error
                result = model.evaluate()
                error_rate = result.get_value()
                # show train loss & error
                plot_train_error(each_batch_id, loss, error_rate)

                # show watched embeddings value.
                plot(params.get_weight(),
                     [(word_dict.get(w, UNK), w) for w in watched_words],
                     canvas=None)

        batch_count = int(len(test_data) / 256) + 1
        avg_loss = 0
        test_evaluator = model.make_evaluator()
        for each_batch_id in xrange(batch_count):
            data_batch = test_data[each_batch_id * batch_size: min(
                (each_batch_id + 1) * batch_size, len(test_data))]
            avg_loss += model.test(data_batch)
            model.evaluate(test_evaluator)

        total_test_error_rate = test_evaluator.get_value()
        avg_loss /= each_batch_id + 1
        plot_test_error(each_epoch_id, avg_loss, total_test_error_rate)


if __name__ == '__main__':
    main()
