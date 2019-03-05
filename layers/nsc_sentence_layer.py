import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier
from layers.hop import hop
from layers.lstm import lstm


def nsc_sentence_layer(x,
                       max_sen_len,
                       max_doc_len,
                       sen_len,
                       identities,
                       hidden_size,
                       emb_dim,
                       sen_hop_cnt=1,
                       bidirectional_lstm=True):
    x = tf.reshape(x, [-1, max_sen_len, x.shape[-1]])
    sen_len = tf.reshape(sen_len, [-1])

    with tf.variable_scope('sentence_layer'):
        # lstm_outputs, _state = lstm(x, sen_len, hidden_size, 'lstm')
        # lstm_outputs = tf.reshape(lstm_outputs, [-1, max_sen_len, hidden_size])
        lstm_bkg, _state = lstm(
            x,
            sen_len,
            hidden_size,
            'lstm_bkg',
            bidirectional=bidirectional_lstm)
        lstm_bkg = tf.reshape(lstm_bkg, [-1, max_sen_len, hidden_size])
        lstm_outputs = lstm_bkg

        sen_bkg = [
            tf.reshape(
                tf.tile(bkg[:, None, :], (1, max_doc_len, 1)), (-1, emb_dim))
            for bkg in identities
        ]
        for ihop in range(sen_hop_cnt):
            last = ihop == sen_hop_cnt - 1
            sen_bkg[0] = hop('hop', last, lstm_outputs, lstm_bkg, sen_bkg[0],
                             sen_bkg[1:], sen_len, max_sen_len, '')
    outputs = tf.reshape(sen_bkg[0], [-1, max_doc_len, hidden_size])

    return outputs
