import paddle
import random


class InMemDataPool(object):
    """
    This is the interface of Paddle DataPool.
    It is an iterator, and will return a batch of data when next invoked.
    The data format is a dictionary which key is data_layer's name, value is the
     mini-batch data.
    """

    def __init__(self, next_data, batch_size, should_shuffle):
        self.__data__ = list(next_data)
        self.__should_shuffle__ = should_shuffle
        self.__idx__ = 0
        self.batch_size = batch_size

    def reset(self):
        self.__idx__ = 0
        if self.__should_shuffle__:
            random.shuffle(self.__data__)

    def next(self):
        if self.__idx__ >= len(self.__data__):
            raise StopIteration()  # end of data.

        begin = self.__idx__
        end = min(self.__idx__ + self.batch_size, len(self.__data__))
        self.__idx__ = end

        retv = dict()
        for k in self.__data__[0].keys():
            retv[k] = list()

        for item in self.__data__[begin: end]:
            for k in item.keys():
                retv[k].append(item[k])

        yield retv

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        # For Python 3
        return self.next()


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
    pass


def plot_train_error(*args):
    """
    Visualize word2vec train error curve.
    :param args:
    :return:
    """
    pass


def plot_test_error(*args):
    """
    Visualize word2vec test error curve.
    :param args:
    :return:
    """
    pass


def prepare_data(dataset, word_dict, unk):
    """
    Convert dataset sentence to Paddle input.
    :param dataset:
    :param word_dict:
    :param unk:
    :return:
    """
    for sentence in dataset:
        sentence = ['<s>', '<s>', '<s>'] + sentence + ['<e>']
        for i in xrange(len(sentence) - 5):
            words = sentence[i:i + 5]
            words = [word_dict.get(w, unk) for w in words]
            retv = dict()
            for i in xrange(len(words)):
                retv['word_%d' % i] = words[i]
            yield retv


def observe(model, optimizer, loss, event):
    """
    Optimizer.train's callback.
    :param optimizer:
    :param model: current training model
    :param loss: current loss value
    :param event: current event, like (on_pass_start, pass_id),
                 (on_batch_end, pass_id, batch_id), etc.
    :return:
    """
    if isinstance(event, paddle.optimizers.event.CompleteTrainBatch):
        if event.batch_id % 100 == 0:  # for every 100 batches
            error_rate = optimizer.evaluate(model)
            plot_train_error(event.pass_id, event.batch_id, loss,
                             error_rate.get_value())
            # plot embedding.
            plot(model.get_parameter(name='embedding'),
                 words=['king', 'queen', 'man', 'woman'], canvas=None)
    elif isinstance(event, paddle.optimizers.event.CompleteTest):
        plot_test_error(event.pass_id, loss, optimizer.evaluate(model))


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

    # consturct word dict.
    word_dict = dataset.get_word_dict(start=START, end=END, unk=UNK, size=1950)

    words = [None] * 5  # reserve a list of 5 elements.

    for i in xrange(len(words)):
        # Network has 5 inputs.
        # which name are word_0, word_1, word_2, word_3, word_4.
        # Size is word_dict length + UNK
        words[i] = paddle.layers.data(name='word_%d' % i,
                                      type=paddle.data.integer(
                                          size=len(word_dict) + 1))

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
    plot(params.get_weight(), watched_words, canvas=None)

    train_data = dataset.train_data(tokenized=True)
    train_data = prepare_data(dataset=train_data, word_dict=word_dict, unk=UNK)
    train_data = InMemDataPool(next_data=train_data, batch_size=256,
                               should_shuffle=True)

    test_data = dataset.test_data(tokenized=True)
    test_data = prepare_data(dataset=test_data, word_dict=word_dict, unk=UNK)
    test_data = InMemDataPool(next_data=test_data, batch_size=256,
                              should_shuffle=False)

    opt = paddle.optimizers.Adam(
        learning_rate=1e-5,
        batch_size=100,
    )

    opt.train(num_epoch=10, train_data=train_data, test_data=test_data,
              observe_callback=observe)


if __name__ == '__main__':
    main()
