from collections import Iterable
import tensorflow as tf
from layers.attention import attention


def hop(scope,
        sentence,
        sentence_bkg,
        bkg_iter,
        bkg_fix,
        doc_len,
        real_max_len,
        convert_flag,
        biases_initializer=tf.initializers.zeros(),
        weights_initializer=tf.contrib.layers.xavier_initializer()):
    """ the param last is no longer used """

    if bkg_fix is None:
        bkg_fix = []
    if not isinstance(bkg_fix, Iterable):
        bkg_fix = [bkg_fix]
    bkg_fix = list(bkg_fix)
    hidden_size = sentence_bkg.shape[2]

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if bkg_iter is not None:
            alphas = attention(
                sentence_bkg, [bkg_iter] + bkg_fix,
                doc_len,
                real_max_len,
                biases_initializer=biases_initializer,
                weights_initializer=weights_initializer)
        else:
            alphas = attention(
                sentence_bkg,
                bkg_fix,
                doc_len,
                real_max_len,
                biases_initializer=biases_initializer,
                weights_initializer=weights_initializer)
        new_bkg = tf.matmul(alphas, sentence)
        new_bkg = tf.reshape(new_bkg, [-1, hidden_size])
        if 'o' in convert_flag:
            new_bkg = bkg_iter + new_bkg
    return new_bkg
