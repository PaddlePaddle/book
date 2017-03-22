#!/bin/env python
import paddle.v2.dataset as dataset
import ntlk

# Cache conll05
dataset.common.download(dataset.conll05.WORDDICT_URL, 'conll05st', \
                        dataset.conll05.WORDDICT_MD5)
dataset.common.download(dataset.conll05.VERBDICT_URL, 'conll05st', \
                        dataset.conll05.VERBDICT_MD5)
dataset.common.download(dataset.conll05.TRGDICT_URL, 'conll05st', \
                        dataset.conll05.TRGDICT_MD5)

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
dataset.common.download(dataset.uci_housing.URL, "uci_housing", dataset.uci_housing.MD5)

# Cache vmt14
dataset.common.download(dataset.vmt14.URL_TRAIN, "wmt14",dataset.vmt14.MD5_TRAIN)
