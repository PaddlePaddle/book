#!/bin/env python
import paddle.dataset as dataset
import nltk

#cifar
dataset.common.download(dataset.cifar.CIFAR100_URL, 'cifar',
                        dataset.cifar.CIFAR100_MD5)
dataset.common.download(dataset.cifar.CIFAR10_URL, 'cifar',
                        dataset.cifar.CIFAR10_MD5)

# Cache conll05
dataset.common.download(dataset.conll05.WORDDICT_URL, 'conll05st', \
                        dataset.conll05.WORDDICT_MD5)
dataset.common.download(dataset.conll05.VERBDICT_URL, 'conll05st', \
                        dataset.conll05.VERBDICT_MD5)
dataset.common.download(dataset.conll05.TRGDICT_URL, 'conll05st', \
                        dataset.conll05.TRGDICT_MD5)
dataset.common.download(dataset.conll05.EMB_URL, 'conll05st',
                        dataset.conll05.EMB_MD5)
dataset.common.download(dataset.conll05.DATA_URL, 'conll05st',
                        dataset.conll05.DATA_MD5)

# Cache imdb
dataset.common.download(dataset.imdb.URL, "imdb", dataset.imdb.MD5)

# Cache imikolov
dataset.common.download(dataset.imikolov.URL, "imikolov", dataset.imikolov.MD5)

# Cache movielens
dataset.common.download('http://files.grouplens.org/datasets/movielens/ml-1m.zip',\
                        'movielens','c4d9eecfca2ab87c1945afe126590906')

# Cache nltk
nltk.download('movie_reviews', download_dir=dataset.common.DATA_HOME)

# Cache uci housing
dataset.common.download(dataset.uci_housing.URL, "uci_housing", \
                        dataset.uci_housing.MD5)

# Cache vmt14
dataset.common.download(dataset.wmt14.URL_TRAIN, "wmt14",\
                        dataset.wmt14.MD5_TRAIN)

#mnist
dataset.common.download(dataset.mnist.TRAIN_IMAGE_URL, 'mnist',
                        dataset.mnist.TRAIN_IMAGE_MD5)
dataset.common.download(dataset.mnist.TRAIN_LABEL_URL, 'mnist',
                        dataset.mnist.TRAIN_LABEL_MD5)
dataset.common.download(dataset.mnist.TEST_IMAGE_URL, 'mnist',
                        dataset.mnist.TEST_IMAGE_MD5)
dataset.common.download(dataset.mnist.TEST_LABEL_URL, 'mnist',
                        dataset.mnist.TEST_LABEL_MD5)
