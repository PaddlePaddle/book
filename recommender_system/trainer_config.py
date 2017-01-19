# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from paddle.trainer_config_helpers import *

try:
    import cPickle as pickle
except ImportError:
    import pickle

is_predict = get_config_arg('is_predict', bool, False)

META_FILE = 'data/meta.bin'

with open(META_FILE, 'rb') as f:
    # load meta file
    meta = pickle.load(f)

if not is_predict:
    define_py_data_sources2(
        'data/train.list',
        'data/test.list',
        module='dataprovider',
        obj='process',
        args={'meta': meta})

settings(
    batch_size=1600, learning_rate=1e-3, learning_method=RMSPropOptimizer())

movie_meta = meta['movie']['__meta__']['raw_meta']
user_meta = meta['user']['__meta__']['raw_meta']

movie_id = data_layer('movie_id', size=movie_meta[0]['max'])
title = data_layer('title', size=len(movie_meta[1]['dict']))
genres = data_layer('genres', size=len(movie_meta[2]['dict']))
user_id = data_layer('user_id', size=user_meta[0]['max'])
gender = data_layer('gender', size=len(user_meta[1]['dict']))
age = data_layer('age', size=len(user_meta[2]['dict']))
occupation = data_layer('occupation', size=len(user_meta[3]['dict']))

embsize = 256

# construct movie feature
movie_id_emb = embedding_layer(input=movie_id, size=embsize)
movie_id_hidden = fc_layer(input=movie_id_emb, size=embsize)

genres_emb = fc_layer(input=genres, size=embsize)

title_emb = embedding_layer(input=title, size=embsize)
title_hidden = text_conv_pool(
    input=title_emb, context_len=5, hidden_size=embsize)

movie_feature = fc_layer(
    input=[movie_id_hidden, title_hidden, genres_emb], size=embsize)

# construct user feature
user_id_emb = embedding_layer(input=user_id, size=embsize)
user_id_hidden = fc_layer(input=user_id_emb, size=embsize)

gender_emb = embedding_layer(input=gender, size=embsize)
gender_hidden = fc_layer(input=gender_emb, size=embsize)

age_emb = embedding_layer(input=age, size=embsize)
age_hidden = fc_layer(input=age_emb, size=embsize)

occup_emb = embedding_layer(input=occupation, size=embsize)
occup_hidden = fc_layer(input=occup_emb, size=embsize)

user_feature = fc_layer(
    input=[user_id_hidden, gender_hidden, age_hidden, occup_hidden],
    size=embsize)

similarity = cos_sim(a=movie_feature, b=user_feature, scale=2)

if not is_predict:
    lbl = data_layer('rating', size=1)
    cost = regression_cost(input=similarity, label=lbl)
    outputs(cost)

else:
    outputs(similarity)
