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
import math
import sys
import argparse
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.nets as nets

IS_SPARSE = True
BATCH_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser("recommender_system")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument(
        '--use_gpu', type=int, default=0, help="Whether to use GPU or not.")
    parser.add_argument(
        '--num_epochs', type=int, default=1, help="number of epochs.")
    args = parser.parse_args()
    return args


def get_usr_combined_features():

    USR_DICT_SIZE = paddle.dataset.movielens.max_user_id() + 1

    uid = layers.data(name='user_id', shape=[1], dtype='int64')

    usr_emb = layers.embedding(
        input=uid,
        dtype='float32',
        size=[USR_DICT_SIZE, 32],
        param_attr='user_table',
        is_sparse=IS_SPARSE)

    usr_fc = layers.fc(input=usr_emb, size=32)

    USR_GENDER_DICT_SIZE = 2

    usr_gender_id = layers.data(name='gender_id', shape=[1], dtype='int64')

    usr_gender_emb = layers.embedding(
        input=usr_gender_id,
        size=[USR_GENDER_DICT_SIZE, 16],
        param_attr='gender_table',
        is_sparse=IS_SPARSE)

    usr_gender_fc = layers.fc(input=usr_gender_emb, size=16)

    USR_AGE_DICT_SIZE = len(paddle.dataset.movielens.age_table)
    usr_age_id = layers.data(name='age_id', shape=[1], dtype="int64")

    usr_age_emb = layers.embedding(
        input=usr_age_id,
        size=[USR_AGE_DICT_SIZE, 16],
        is_sparse=IS_SPARSE,
        param_attr='age_table')

    usr_age_fc = layers.fc(input=usr_age_emb, size=16)

    USR_JOB_DICT_SIZE = paddle.dataset.movielens.max_job_id() + 1
    usr_job_id = layers.data(name='job_id', shape=[1], dtype="int64")

    usr_job_emb = layers.embedding(
        input=usr_job_id,
        size=[USR_JOB_DICT_SIZE, 16],
        param_attr='job_table',
        is_sparse=IS_SPARSE)

    usr_job_fc = layers.fc(input=usr_job_emb, size=16)

    concat_embed = layers.concat(
        input=[usr_fc, usr_gender_fc, usr_age_fc, usr_job_fc], axis=1)

    usr_combined_features = layers.fc(input=concat_embed, size=200, act="tanh")

    return usr_combined_features


def get_mov_combined_features():

    MOV_DICT_SIZE = paddle.dataset.movielens.max_movie_id() + 1

    mov_id = layers.data(name='movie_id', shape=[1], dtype='int64')

    mov_emb = layers.embedding(
        input=mov_id,
        dtype='float32',
        size=[MOV_DICT_SIZE, 32],
        param_attr='movie_table',
        is_sparse=IS_SPARSE)

    mov_fc = layers.fc(input=mov_emb, size=32)

    CATEGORY_DICT_SIZE = len(paddle.dataset.movielens.movie_categories())

    category_id = layers.data(
        name='category_id', shape=[1], dtype='int64', lod_level=1)

    mov_categories_emb = layers.embedding(
        input=category_id, size=[CATEGORY_DICT_SIZE, 32], is_sparse=IS_SPARSE)

    mov_categories_hidden = layers.sequence_pool(
        input=mov_categories_emb, pool_type="sum")

    MOV_TITLE_DICT_SIZE = len(paddle.dataset.movielens.get_movie_title_dict())

    mov_title_id = layers.data(
        name='movie_title', shape=[1], dtype='int64', lod_level=1)

    mov_title_emb = layers.embedding(
        input=mov_title_id, size=[MOV_TITLE_DICT_SIZE, 32], is_sparse=IS_SPARSE)

    mov_title_conv = nets.sequence_conv_pool(
        input=mov_title_emb,
        num_filters=32,
        filter_size=3,
        act="tanh",
        pool_type="sum")

    concat_embed = layers.concat(
        input=[mov_fc, mov_categories_hidden, mov_title_conv], axis=1)

    mov_combined_features = layers.fc(input=concat_embed, size=200, act="tanh")

    return mov_combined_features


def inference_program():
    usr_combined_features = get_usr_combined_features()
    mov_combined_features = get_mov_combined_features()

    inference = layers.cos_sim(X=usr_combined_features, Y=mov_combined_features)
    scale_infer = layers.scale(x=inference, scale=5.0)

    label = layers.data(name='score', shape=[1], dtype='float32')
    square_cost = layers.square_error_cost(input=scale_infer, label=label)
    avg_cost = layers.mean(square_cost)

    return scale_infer, avg_cost


def optimizer_func():
    return fluid.optimizer.SGD(learning_rate=0.2)


def train(use_cuda, params_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    if args.enable_ce:
        train_reader = paddle.batch(
            paddle.dataset.movielens.train(), batch_size=BATCH_SIZE)
        test_reader = paddle.batch(
            paddle.dataset.movielens.test(), batch_size=BATCH_SIZE)
    else:
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.movielens.train(), buf_size=8192),
            batch_size=BATCH_SIZE)
        test_reader = paddle.batch(
            paddle.dataset.movielens.test(), batch_size=BATCH_SIZE)

    feed_order = [
        'user_id', 'gender_id', 'age_id', 'job_id', 'movie_id', 'category_id',
        'movie_title', 'score'
    ]

    main_program = fluid.default_main_program()
    star_program = fluid.default_startup_program()
    if args.enable_ce:
        main_program.random_seed = 90
        star_program.random_seed = 90

    scale_infer, avg_cost = inference_program()

    test_program = main_program.clone(for_test=True)
    sgd_optimizer = optimizer_func()
    sgd_optimizer.minimize(avg_cost)
    exe = fluid.Executor(place)

    def train_test(program, reader):
        count = 0
        feed_var_list = [
            program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
        test_exe = fluid.Executor(place)
        accumulated = 0
        for test_data in reader():
            avg_cost_np = test_exe.run(
                program=program,
                feed=feeder_test.feed(test_data),
                fetch_list=[avg_cost])
            accumulated += avg_cost_np[0]
            count += 1
        return accumulated / count

    def train_loop():
        feed_list = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list, place)
        exe.run(star_program)

        for pass_id in range(PASS_NUM):
            for batch_id, data in enumerate(train_reader()):
                # train a mini-batch
                outs = exe.run(
                    program=main_program,
                    feed=feeder.feed(data),
                    fetch_list=[avg_cost])
                out = np.array(outs[0])

                # get test avg_cost
                test_avg_cost = train_test(test_program, test_reader)

                # if test_avg_cost < 4.0: # Change this number to adjust accuracy
                if batch_id == 20:

                    if args.enable_ce:
                        print("kpis\ttest_cost\t%f" % float(test_avg_cost))

                    if params_dirname is not None:
                        fluid.io.save_inference_model(params_dirname, [
                            "user_id", "gender_id", "age_id", "job_id",
                            "movie_id", "category_id", "movie_title"
                        ], [scale_infer], exe)
                    return
                print('EpochID {0}, BatchID {1}, Test Loss {2:0.2}'.format(
                    pass_id + 1, batch_id + 1, float(test_avg_cost)))

                if math.isnan(float(out[0])):
                    sys.exit("got NaN loss, training failed.")

    train_loop()


def infer(use_cuda, params_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    # Use the first data from paddle.dataset.movielens.test() as input.
    # Use create_lod_tensor(data, lod, place) API to generate LoD Tensor,
    # where `data` is a list of sequences of index numbers, `lod` is
    # the level of detail (lod) info associated with `data`.
    # For example, data = [[10, 2, 3], [2, 3]] means that it contains
    # two sequences of indexes, of length 3 and 2, respectively.
    # Correspondingly, lod = [[3, 2]] contains one level of detail info,
    # indicating that `data` consists of two sequences of length 3 and 2.
    infer_movie_id = 783
    infer_movie_name = paddle.dataset.movielens.movie_info()[
        infer_movie_id].title

    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()

    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inferencer, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

        # Use the first data from paddle.dataset.movielens.test() as input
        assert feed_target_names[0] == "user_id"
        # Use create_lod_tensor(data, recursive_sequence_lengths, place) API
        # to generate LoD Tensor where `data` is a list of sequences of index
        # numbers, `recursive_sequence_lengths` is the length-based level of detail
        # (lod) info associated with `data`.
        # For example, data = [[10, 2, 3], [2, 3]] means that it contains
        # two sequences of indexes, of length 3 and 2, respectively.
        # Correspondingly, recursive_sequence_lengths = [[3, 2]] contains one
        # level of detail info, indicating that `data` consists of two sequences
        # of length 3 and 2, respectively.
        user_id = fluid.create_lod_tensor([[np.int64(1)]], [[1]], place)

        assert feed_target_names[1] == "gender_id"
        gender_id = fluid.create_lod_tensor([[np.int64(1)]], [[1]], place)

        assert feed_target_names[2] == "age_id"
        age_id = fluid.create_lod_tensor([[np.int64(0)]], [[1]], place)

        assert feed_target_names[3] == "job_id"
        job_id = fluid.create_lod_tensor([[np.int64(10)]], [[1]], place)

        assert feed_target_names[4] == "movie_id"
        movie_id = fluid.create_lod_tensor([[np.int64(783)]], [[1]], place)

        assert feed_target_names[5] == "category_id"
        category_id = fluid.create_lod_tensor(
            [np.array([10, 8, 9], dtype='int64')], [[3]], place)

        assert feed_target_names[6] == "movie_title"
        movie_title = fluid.create_lod_tensor(
            [np.array([1069, 4140, 2923, 710, 988], dtype='int64')], [[5]],
            place)

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        results = exe.run(
            inferencer,
            feed={
                feed_target_names[0]: user_id,
                feed_target_names[1]: gender_id,
                feed_target_names[2]: age_id,
                feed_target_names[3]: job_id,
                feed_target_names[4]: movie_id,
                feed_target_names[5]: category_id,
                feed_target_names[6]: movie_title
            },
            fetch_list=fetch_targets,
            return_numpy=False)
        predict_rating = np.array(results[0])
        print("Predict Rating of user id 1 on movie \"" + infer_movie_name +
              "\" is " + str(predict_rating[0][0]))
        print("Actual Rating of user id 1 on movie \"" + infer_movie_name +
              "\" is 4.")


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    params_dirname = "recommender_system.inference.model"
    train(use_cuda=use_cuda, params_dirname=params_dirname)
    infer(use_cuda=use_cuda, params_dirname=params_dirname)


if __name__ == '__main__':
    args = parse_args()
    PASS_NUM = args.num_epochs
    use_cuda = args.use_gpu
    main(use_cuda)
