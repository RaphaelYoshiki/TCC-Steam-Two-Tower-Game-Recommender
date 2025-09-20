# two_tower_model.py
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Concatenate, Dropout,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, TimeDistributed, Lambda
)
from tensorflow.keras.models import Model

def create_two_tower_model(user_max_lengths, game_max_lengths, embedding_dim,
                           unique_game_devs, unique_game_gens, unique_game_cats, unique_game_tags):

    # list_size desconhecido, usamos None
    list_size = None

    # ---------- Inputs usuário ----------
    ui_genres = Input(shape=(list_size, user_max_lengths['interacted_genres']),
                      dtype='int32', name='user_interacted_genres')
    ui_cats   = Input(shape=(list_size, user_max_lengths['interacted_categories']),
                      dtype='int32', name='user_interacted_categories')
    ui_tags   = Input(shape=(list_size, user_max_lengths['interacted_tags']),
                      dtype='int32', name='user_interacted_tags')
    ui_devs   = Input(shape=(list_size, user_max_lengths['interacted_developers']),
                      dtype='int32', name='user_interacted_developers')

    # ---------- Inputs jogo ----------
    g_genres = Input(shape=(list_size, game_max_lengths['genre_ids']),
                     dtype='int32', name='game_genre_ids')
    g_cats   = Input(shape=(list_size, game_max_lengths['category_ids']),
                     dtype='int32', name='game_category_ids')
    g_tags   = Input(shape=(list_size, game_max_lengths['user_tag_ids']),
                     dtype='int32', name='game_user_tag_ids')
    g_devs   = Input(shape=(list_size, game_max_lengths['developer_ids']),
                     dtype='int32', name='game_developer_ids')
    g_review = Input(shape=(list_size, 1), dtype='float32', name='game_review_score')

    # ---------- Embeddings ----------
    genre_embedding = Embedding(input_dim=max(1, unique_game_gens), output_dim=embedding_dim, mask_zero=False, name='emb_genre')
    cat_embedding   = Embedding(input_dim=max(1, unique_game_cats), output_dim=embedding_dim, mask_zero=False, name='emb_cat')
    tag_embedding   = Embedding(input_dim=max(1, unique_game_tags), output_dim=embedding_dim, mask_zero=False, name='emb_tag')
    dev_embedding   = Embedding(input_dim=max(1, unique_game_devs), output_dim=embedding_dim, mask_zero=False, name='emb_dev')

    def seq_encode(x, emb):
        # x: (batch, list_size, seq_len)
        e = emb(x)  # (batch, list_size, seq_len, emb_dim)  emb é compartilhado, não cria variáveis novas
        avg = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling1D())(e)
        mx  = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalMaxPooling1D())(e)
        concat = Concatenate(axis=-1)([avg, mx])  # (batch, list_size, 2*emb_dim)
        return concat

    # ---------- User tower ----------
    u_gen = seq_encode(ui_genres, genre_embedding)
    u_cat = seq_encode(ui_cats, cat_embedding)
    u_tag = seq_encode(ui_tags, tag_embedding)
    u_dev = seq_encode(ui_devs, dev_embedding)

    user_concat = Concatenate(axis=-1)([u_gen, u_cat, u_tag, u_dev])
    user_dense = TimeDistributed(Dense(256, activation='relu'))(user_concat)
    user_dense = TimeDistributed(Dropout(0.3))(user_dense)
    user_dense = TimeDistributed(Dense(128, activation='relu'))(user_dense)
    user_dense = TimeDistributed(Dropout(0.2))(user_dense)
    user_emb = TimeDistributed(Dense(embedding_dim, activation=None), name='user_embedding')(user_dense)
    # (batch, list_size, emb_dim)

    # ---------- Game tower ----------
    g_gen = seq_encode(g_genres, genre_embedding)
    g_cat = seq_encode(g_cats, cat_embedding)
    g_tag = seq_encode(g_tags, tag_embedding)
    g_dev = seq_encode(g_devs, dev_embedding)

    game_concat = Concatenate(axis=-1)([g_gen, g_cat, g_tag, g_dev, g_review])
    game_dense = TimeDistributed(Dense(128, activation='relu'))(game_concat)
    game_dense = TimeDistributed(Dropout(0.3))(game_dense)
    game_dense = TimeDistributed(Dense(128, activation='relu'))(game_dense)
    game_dense = TimeDistributed(Dropout(0.2))(game_dense)
    game_emb = TimeDistributed(Dense(embedding_dim, activation=None), name='game_embedding')(game_dense)
    # (batch, list_size, emb_dim)

    # ---------- Score ----------
    dot = tf.reduce_sum(user_emb * game_emb, axis=-1)  # (batch, list_size)
    output = dot  # (batch, list_size)

    model = Model(
        inputs=[ui_genres, ui_cats, ui_tags, ui_devs,
                g_genres, g_cats, g_tags, g_devs, g_review],
        outputs=output,
        name='two_tower_model'
    )

    return model
