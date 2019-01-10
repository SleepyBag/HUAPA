import tensorflow as tf
from tensorflow import constant as const
from colored import stylize, fg
from math import sqrt
import numpy as np
import capslayer as cl
lookup = tf.nn.embedding_lookup


def lstm(inputs, sequence_length, hidden_size, scope):
    cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_size // 2, initializer=tf.contrib.layers.xavier_initializer())
    cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_size // 2, initializer=tf.contrib.layers.xavier_initializer())
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw, cell_bw=cell_bw, inputs=inputs,
        sequence_length=sequence_length, dtype=tf.float32, scope=scope)
    outputs = tf.concat(outputs, axis=2)
    return outputs, state


def var(name, shape, initializer):
    return tf.get_variable(name, shape=shape, initializer=initializer)


class CSC(object):

    def __init__(self, args):
        self.max_doc_len = args['max_doc_len']
        self.max_sen_len = args['max_sen_len']
        self.cls_cnt = args['cls_cnt']
        self.embedding = args['embedding']
        self.emb_dim = args['emb_dim']
        self.hidden_size = args['hidden_size']
        self.l2_rate = args['l2_rate']
        self.sen_aspect_cnt = args['sen_aspect_cnt']
        self.doc_aspect_cnt = args['doc_aspect_cnt']

        self.best_dev_acc = .0
        self.best_test_acc = .0
        self.best_test_rmse = .0

        # initializers for parameters
        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.biases_initializer = tf.initializers.zeros()
        self.emb_initializer = tf.contrib.layers.xavier_initializer()

        hsize = self.hidden_size

        # embeddings in the model
        with tf.variable_scope('emb'):
            self.embeddings = {
                'wrd_emb': const(self.embedding, name='wrd_emb', dtype=tf.float32),
                # 'wrd_emb': tf.Variable(self.embedding, name='wrd_emb', dtype=tf.float32),
            }

    def get_weight(self, name, shape):
        return var(name, shape, self.weights_initializer)

    def get_bias(self, name, shape):
        return var(name, shape, self.biases_initializer)

    def csc(self, x, max_sen_len, max_doc_len, sen_len, doc_len):
        # spilt each batch into max_doc_len ones
        sen_x = tf.reshape(x, [-1, max_sen_len, self.emb_dim], name='sentence_x')
        sen_len = tf.reshape(sen_len, [-1], name='sen_len')

        hidden_size = self.hidden_size

        # lstm_output should be tensor with shape
        # [batch_size*max_doc_len, max_sen_len, hidden_size]
        # here a real batch is considered as max_doc_len batches
        lstm_output, _state = lstm(sen_x, sen_len, self.hidden_size, 'embedding_to_lstm')
        lstm_output = tf.reshape(lstm_output, [-1, max_sen_len, self.hidden_size, 1],
                                 name='lstm_output')

        # caps_sen should be tensor with shape
        # [batch_size*max_doc_len, aspect_cnt, d, 1]
        # here a real batch is considered as max_doc_len batches
        activation = tf.zeros([1, 1])
        # lstm_output = sen_x[:, :, :, None]
        caps_sen, activation = cl.layers.dense(inputs=lstm_output,
                                               activation=activation,
                                               num_outputs=self.sen_aspect_cnt,
                                               out_caps_dims=[hidden_size, 1],
                                               sequence_length=sen_len,
                                               max_length=max_sen_len,
                                               share=True,
                                               transform=False,
                                               routing_method='DynamicRouting',
                                               name='lstm_to_caps')

        caps_sen = caps_sen[:, :self.sen_aspect_cnt, :, :]
        caps_sen = tf.reshape(caps_sen, [-1, max_doc_len * self.sen_aspect_cnt, hidden_size, 1], name='caps_sen')

        activation = tf.reshape(activation, [-1, max_doc_len * self.sen_aspect_cnt])
        # caps_doc should be tensor with shape
        # [batch_size, aspect_cnt, d, 1]
        # here a real batch is considered as max_doc_len batches
        caps_doc, activation = cl.layers.dense(inputs=caps_sen,
                                               activation=activation,
                                               num_outputs=self.doc_aspect_cnt,
                                               out_caps_dims=[hidden_size, 1],
                                               sequence_length=doc_len * self.sen_aspect_cnt,
                                               max_length=max_doc_len * self.sen_aspect_cnt,
                                               share=True,
                                               transform=True,
                                               routing_method='DynamicRouting',
                                               name='sen_to_doc')

        caps_doc = caps_doc[:, :self.doc_aspect_cnt, :, :]

        caps_doc = tf.layers.flatten(caps_doc)
        outputs = tf.layers.dense(caps_doc, self.cls_cnt)

        return outputs

    def build(self, data_iter):
        # get the inputs
        with tf.variable_scope('inputs'):
            input_map = data_iter.get_next()
            input_x, input_y, sen_len, doc_len = \
                (input_map['content'], input_map['rating'],
                 input_map['sen_len'], input_map['doc_len'])

            input_x = lookup(self.embeddings['wrd_emb'], input_x, name='cur_wrd_embedding')

        # build the process of model
        d_hat = self.csc(input_x, self.max_sen_len, self.max_doc_len,
                         sen_len, doc_len)
        prediction = tf.argmax(d_hat, 1, name='prediction')

        with tf.variable_scope("loss"):
            sce = tf.nn.softmax_cross_entropy_with_logits_v2
            self.loss = sce(logits=d_hat, labels=tf.one_hot(input_y, self.cls_cnt))

            regularizer = tf.zeros(1)
            params = tf.trainable_variables()
            for param in params:
                if param not in self.embeddings.values():
                    regularizer += tf.nn.l2_loss(param)
            self.loss = tf.reduce_mean(self.loss) + self.l2_rate * regularizer

        with tf.variable_scope("metrics"):
            correct_prediction = tf.equal(prediction, input_y)
            mse = tf.reduce_sum(tf.square(prediction - input_y), name="mse")
            correct_num = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.int32), name="correct_num")
            accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"), name="accuracy")

        return self.loss, mse, correct_num, accuracy

    def output_metrics(self, metrics, data_length):
        loss, mse, correct_num, accuracy = metrics
        info = 'Loss = %.3f, RMSE = %.3f, Acc = %.3f' % \
            (loss / data_length, sqrt(float(mse) / data_length), float(correct_num) / data_length)
        return info

    def record_metrics(self, dev_metrics, test_metrics, devlen, testlen):
        _dev_loss, dev_mse, dev_correct_num, dev_accuracy = dev_metrics
        _test_loss, test_mse, test_correct_num, test_accuracy = test_metrics
        dev_accuracy = float(dev_correct_num) / devlen
        test_accuracy = float(test_correct_num) / testlen
        test_rmse = sqrt(float(test_mse) / testlen)
        if dev_accuracy > self.best_dev_acc:
            self.best_dev_acc = dev_accuracy
            self.best_test_acc = test_accuracy
            self.best_test_rmse = test_rmse
            info = 'NEW best dev acc: %.3f, NEW best test acc: %.3f, NEW best test RMSE: %.3f' % \
                (self.best_dev_acc, self.best_test_acc, self.best_test_rmse)
        else:
            info = 'best dev acc: %.3f, best test acc: %.3f, best test RMSE: %.3f' % \
                (self.best_dev_acc, self.best_test_acc, self.best_test_rmse)
        return info

    def train(self, optimizer, global_step):
        grads_and_vars = optimizer.compute_gradients(self.loss)

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        return train_op
