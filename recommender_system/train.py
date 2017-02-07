from paddle import layers
from paddle import model
from paddle import optimizer

from paddle.datasets import movielens


def main():

    paddle.init(use_gpu = True)

    # Keras
    X_train, y_train, X_test, y_test = movielens.load_data('xxx')

    embsize = 256

    # It's better if we can refactor network, but I did not figure out now.

    # construct movie feature
    mv_id = data_layer('mv_id', size = xxx))
    title = data_layer('title', size = xxx))
    geres = data_layer('geres', size = xxx))

    movie_id_emb = embedding_layer(input=mv_id, size=embsize))
    movie_id_hid = fc_layer(input=movie_id_emb, size=embsize))
    genres_embed = fc_layer(input=geres, size=embsize))
    titles_embed = embedding_layer(input=title, size=embsize)）
    title_hidden = text_conv_pool(input=titles_embed, context_len=5, hidden_size=embsize))
    movie_feature = fc_layer(input=[movie_id_hid, title_hidden, genres_embed], size=embsize))
    
    # construct user feature
    user_id = data_layer(size = xxx))
    gender = data_layer(size = xxx))
    age = data_layer(size = xxx))
    occupation = data_layer(size = xxx))

    user_id_emb = embedding_layer(input=user_id, size=embsize))
    user_id_hidden = fc_layer(input=user_id_emb, size=embsize))
    gender_emb = embedding_layer(input=gender, size=embsize))
    gender_hidden = fc_layer(input=gender_emb, size=embsize))
    age_emb = embedding_layer(input=age, size=embsize))
    age_hidden = fc_layer(input=age_emb, size=embsize))
    occup_emb = embedding_layer(input=occupation, size=embsize))
    occup_hidden = fc_layer(input=occup_emb, size=embsize))
    user_feature = fc_layer(
        input=[user_id_hidden, gender_hidden, age_hidden, occup_hidden], size=embsize)

    similarity = cos_sim(a=movie_feature, b=user_feature, scale=2))
    lbl = data_layer('rating', size=1))
    cost = regression_cost(input=similarity, label=lbl))

    # how to refactor and line up network to make it more readable and elegant
    model.network(cost)

    # optimization
    model.optimizer(RMSprop(batch_size = 1600, learning_rate = 1e-3))

    # can we make it more precise? keras only use a fit function in here
    model.train(X_train, y_train, plot=True)
    model.test(X_test, y_test, plot=True)

if __name__ == '__main__':
    main()