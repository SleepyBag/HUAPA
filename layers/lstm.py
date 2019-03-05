import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier


def lstm(inputs, sequence_length, hidden_size, scope, bidirectional=True):
    if bidirectional:
        cell_fw = tf.nn.rnn_cell.LSTMCell(
            hidden_size // 2, forget_bias=0., initializer=xavier())
        cell_bw = tf.nn.rnn_cell.LSTMCell(
            hidden_size // 2, forget_bias=0., initializer=xavier())
        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32,
            scope=scope)
        outputs = tf.concat(outputs, axis=2)
    else:
        cell = tf.nn.rnn_cell.LSTMCell(
            hidden_size, forget_bias=0., initializer=xavier())
        outputs, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32,
            scope=scope)
        outputs = tf.concat(outputs, axis=2)
    return outputs, state
