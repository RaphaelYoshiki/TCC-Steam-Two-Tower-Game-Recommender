import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Dropout, TimeDistributed, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def create_two_tower_model(user_max_lengths, game_max_lengths, embedding_dim,
                           unique_game_gens, unique_game_cats, unique_game_tags, unique_langs, top_n=5):
    list_size = None
    emb_reg = l2(1e-2)
    dense_reg = l2(1e-3)

    # ---------- Inputs usuário ----------
    u_lang = Input(shape=(list_size,), dtype='int32', name='user_lang_id')
    u_top_genre = Input(shape=(list_size,), dtype='int32', name='user_top_genre')
    u_top_cat = Input(shape=(list_size,), dtype='int32', name='user_top_cat')
    u_top_tag = Input(shape=(list_size,), dtype='int32', name='user_top_tag')
    u_genre_dominance = Input(shape=(list_size, 1), dtype='float32', name='user_genre_dominance')
    u_genre_diversity = Input(shape=(list_size, 1), dtype='float32', name='user_genre_diversity')

    # ---------- Inputs jogo ----------
    g_genres = Input(shape=(list_size, game_max_lengths['genre_ids']), dtype='int32', name='game_genre_ids')
    g_cats = Input(shape=(list_size, game_max_lengths['category_ids']), dtype='int32', name='game_category_ids')
    g_tags = Input(shape=(list_size, game_max_lengths['user_tag_ids']), dtype='int32', name='game_user_tag_ids')
    g_review = Input(shape=(list_size, 1), dtype='float32', name='game_review_score')

    # ---------- Embeddings ----------
    lang_emb = Embedding(unique_langs, embedding_dim // 2, name='emb_lang', embeddings_regularizer=emb_reg)
    top_genre_emb = Embedding(unique_game_gens, embedding_dim // 4, name='emb_top_genre')
    top_cat_emb = Embedding(unique_game_cats, embedding_dim // 4, name='emb_top_cat') 
    top_tag_emb = Embedding(unique_game_tags, embedding_dim // 4, name='emb_top_tag')
    
    u_top_genre_emb = top_genre_emb(u_top_genre)
    u_top_cat_emb = top_cat_emb(u_top_cat)
    u_top_tag_emb = top_tag_emb(u_top_tag)
    
    genre_emb = Embedding(unique_game_gens, embedding_dim, name='emb_genre', embeddings_regularizer=emb_reg)
    cat_emb = Embedding(unique_game_cats, embedding_dim, name='emb_cat', embeddings_regularizer=emb_reg)
    tag_emb = Embedding(unique_game_tags, embedding_dim, name='emb_tag', embeddings_regularizer=emb_reg)

    def seq_avg(x, emb):
        e = emb(x)
        mask = tf.cast(tf.not_equal(x, 0), tf.float32)
        mask_expanded = tf.expand_dims(mask, axis=-1)  # Equivalente a [..., None]
        return tf.reduce_sum(e * mask_expanded, axis=2) / (tf.reduce_sum(mask_expanded, axis=2) + 1e-6)

    # ---------- Torre do usuário ----------
    u_lang_emb = lang_emb(u_lang)

    user_concat = Concatenate(axis=-1)([
        u_lang_emb, u_top_genre_emb, u_top_cat_emb, 
        u_top_tag_emb, u_genre_dominance, u_genre_diversity
    ])

    user_dense = Dense(128, activation='relu', kernel_regularizer=dense_reg)(user_concat)
    user_dense = BatchNormalization()(user_dense)
    user_dense = Dropout(0.4)(user_dense)
    user_dense = Dense(64, activation='relu', kernel_regularizer=dense_reg)(user_dense)
    user_dense = Dropout(0.3)(user_dense)
    user_emb = Dense(embedding_dim, activation=None, kernel_regularizer=dense_reg)(user_dense)
    
    # ---------- Torre do jogo ----------
    g_gen_emb = seq_avg(g_genres, genre_emb)
    g_cat_emb = seq_avg(g_cats, cat_emb)
    g_tag_emb = seq_avg(g_tags, tag_emb)
    game_concat = Concatenate(axis=-1)([g_gen_emb, g_cat_emb, g_tag_emb, g_review])

    game_dense = TimeDistributed(Dense(128, activation='relu', kernel_regularizer=dense_reg))(game_concat)
    game_dense = BatchNormalization()(game_dense)
    game_dense = TimeDistributed(Dropout(0.5))(game_dense)
    game_dense = TimeDistributed(Dense(64, activation='relu', kernel_regularizer=dense_reg))(game_dense)
    game_dense = TimeDistributed(Dropout(0.4))(game_dense)
    game_emb = TimeDistributed(Dense(embedding_dim, activation=None, kernel_regularizer=dense_reg))(game_dense)
    
    # ---------- Similaridade e saídas ----------
    dot = tf.reduce_sum(user_emb * game_emb, axis=-1)
    score_out = tf.keras.layers.Activation('linear', name='score')(dot)

    model = Model(
        inputs=[u_lang, u_top_genre, 
                u_top_cat, u_top_tag,
                u_genre_dominance, u_genre_diversity,
                g_genres, g_cats, 
                g_tags, g_review],
        outputs={'score': score_out},
        name='two_tower_steam'
    )

    return model
