import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier
from layers.hop import hop
from layers.lstm import lstm


def nsc_document_layer(x,
                       max_doc_len,
                       doc_len,
                       identities,
                       hidden_size,
                       doc_hop_cnt=1,
                       bidirectional_lstm=True):

    with tf.variable_scope('document_layer'):
        # lstm_outputs, _state = lstm(outputs, doc_len, hidden_size, 'lstm')
        lstm_bkg, _state = lstm(x, doc_len, hidden_size, 'lstm_bkg', bidirectional=bidirectional_lstm)
        lstm_outputs = lstm_bkg

        doc_bkg = [i for i in identities]
        for ihop in range(doc_hop_cnt):
            last = ihop == doc_hop_cnt - 1
            doc_bkg[0] = hop('hop', last, lstm_outputs, lstm_bkg, doc_bkg[0],
                             doc_bkg[1:], doc_len, max_doc_len, '')
    outputs = doc_bkg[0]

    return outputs
