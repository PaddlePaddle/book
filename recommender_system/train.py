import paddle.layer
import paddle.model
import paddle.optimizer
import paddle.datasets.movielens


def main():

    paddle.init(use_gpu = True)

    # Keras
    (X_train, y_train), (X_test, y_test) = paddle.datasets.movielens.load_data('xxx')

    embsize = 256

    # It's better if we can refactor network, but I did not figure out now.

    # construct movie feature
    mv_id = paddle.layer.data('mv_id', size = xxx)
    title = paddle.layer.data('title', size = xxx)
    geres = paddle.layer.data('geres', size = xxx)

    movie_id_emb = paddle.layer.embedding(input=mv_id, size=embsize)
    movie_id_hid = paddle.layer.fc(input=movie_id_emb, size=embsize)
    genres_embed = paddle.layer.fc(input=geres, size=embsize)
    titles_embed = paddle.layer.embedding(input=title, size=embsize)
    title_hidden = paddle.layer.text_conv_pool(input=titles_embed, context_len=5, hidden_size=embsize))
    movie_feture = paddle.layer.fc_layer(input=[movie_id_hid, title_hidden, genres_embed], size=embsize))
    
    # construct user feature
    user_id = paddle.layer.data(size = xxx))
    gender = paddle.layer.data(size = xxx))
    age  = paddle.layer.data(size = xxx))
    occupation = paddle.layer.data(size = xxx))

    user_id_emb = paddle.layer.embedding(input=user_id, size=embsize)
    user_id_hid = paddle.layer.fc(input=user_id_emb, size=embsize)

    gender_embd = paddle.layer.embedding(input=gender, size=embsize)
    gender_hidd = paddle.layer.fc(input=gender_embd, size=embsize)

    age_embding = paddle.layer.embedding(input=age, size=embsize)
    age_hiddens = paddle.layer.fc(input=age_embding, size=embsize)

    occup_embed = paddle.layer.embedding(input=occupation, size=embsize)
    occup_hiden = paddle.layer.fc(input=occup_embed, size=embsize)

    user_feture = paddle.layer.fc(
        input=[user_id_hid, gender_hidd, age_hidden, occup_hiden], size=embsize)
    
    sim = paddle.layer.cos_sim(a=movie_feature, b=user_feature, scale=2)

    model = paddle.model.create(sim)

    lbl = paddle.layer.data('rating', size=1)
    RMSprop.train(model, paddle.cost.regression(sim, lbl), 
                  batch_size = 1600, learning_rate = 1e-3, ...)




if __name__ == '__main__':
    main()