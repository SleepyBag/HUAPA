from math import sqrt
import tensorflow as tf
from tensorflow import constant as const
from layers.nsc_sentence_layer import nsc_sentence_layer
from layers.nsc_document_layer import nsc_document_layer
from layers.lstm import lstm
lookup = tf.nn.embedding_lookup


def var(name, shape, initializer):
    return tf.get_variable(name, shape=shape, initializer=initializer)


class AMHNSC_POLARITY(object):
    def __init__(self, args):
        self.max_doc_len = args['max_doc_len']
        self.max_sen_len = args['max_sen_len']
        self.cls_cnt = args['cls_cnt']
        self.embedding = args['embedding']
        self.emb_dim = args['emb_dim']
        self.hidden_size = args['hidden_size']
        self.usr_cnt = args['usr_cnt']
        self.prd_cnt = args['prd_cnt']
        self.sen_hop_cnt = args['sen_hop_cnt']
        self.doc_hop_cnt = args['doc_hop_cnt']
        self.l2_rate = args['l2_rate']
        self.convert_flag = ''
        self.debug = args['debug']
        self.lambda1 = args['lambda1']
        self.lambda2 = args['lambda2']
        self.lambda3 = args['lambda3']
        self.embedding_lr = args['embedding_lr']

        self.best_dev_acc = .0
        self.best_test_acc = .0
        self.best_test_rmse = .0

        # initializers for parameters
        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.biases_initializer = tf.initializers.zeros()
        self.emb_initializer = tf.contrib.layers.xavier_initializer()

        # embeddings in the model
        with tf.variable_scope('emb'):
            self.embeddings = {
                'wrd_emb':
                const(self.embedding, name='wrd_emb', dtype=tf.float32),
                #  tf.Variable(self.embedding, name='wrd_emb', dtype=tf.float32),
                'usr_emb':
                var('usr_emb', [self.usr_cnt, self.emb_dim],
                    self.emb_initializer),
                'prd_emb':
                var('prd_emb', [self.prd_cnt, self.emb_dim],
                    self.emb_initializer)
            }

        # for tensorboard
        if self.debug:
            tf.summary.histogram('usr_emb', self.embeddings['usr_emb'])
            tf.summary.histogram('prd_emb', self.embeddings['prd_emb'])

    def build(self, data_iter):
        # get the inputs
        with tf.variable_scope('inputs'):
            input_map = data_iter.get_next()
            usrid, prdid, input_x, input_y, sen_len, doc_len, polarity = \
                (input_map['usr'], input_map['prd'],
                 input_map['content'], input_map['rating'],
                 input_map['sen_len'], input_map['doc_len'],
                 input_map['polarity'])

            usr = lookup(
                self.embeddings['usr_emb'], usrid, name='cur_usr_embedding')
            prd = lookup(
                self.embeddings['prd_emb'], prdid, name='cur_prd_embedding')
            input_x = lookup(
                self.embeddings['wrd_emb'], input_x, name='cur_wrd_embedding')
            input_x = tf.reshape(
                input_x,
                [-1, self.max_doc_len, self.max_sen_len, self.emb_dim])
            sen_len = tf.reshape(sen_len, [-1, self.max_doc_len])
            polarity = tf.reshape(polarity,
                                  [-1, self.max_doc_len, self.max_sen_len])
            nscua_input_x, nscpa_input_x = input_x, input_x

        # padding content with user embedding
        tiled_usr = tf.layers.dense(
            usr,
            self.emb_dim,
            use_bias=False,
            kernel_initializer=self.weights_initializer,
            bias_initializer=self.biases_initializer)
        tiled_prd = tf.layers.dense(
            prd,
            self.emb_dim,
            use_bias=False,
            kernel_initializer=self.weights_initializer,
            bias_initializer=self.biases_initializer)
        tiled_usr = tf.tile(tiled_usr[:, None, None, :],
                            [1, self.max_doc_len, 1, 1])
        tiled_prd = tf.tile(tiled_prd[:, None, None, :],
                            [1, self.max_doc_len, 1, 1])
        nscua_input_x = tf.concat([tiled_usr, nscua_input_x], axis=2)
        nscpa_input_x = tf.concat([tiled_prd, nscpa_input_x], axis=2)
        aug_sen_len = tf.where(
            tf.equal(sen_len, 0), tf.zeros_like(sen_len), sen_len + 1)
        self.max_sen_len += 1
        nscua_input_x = tf.reshape(nscua_input_x,
                                   [-1, self.max_sen_len, self.emb_dim])
        nscpa_input_x = tf.reshape(nscpa_input_x,
                                   [-1, self.max_sen_len, self.emb_dim])

        # build the process of model
        sen_embs, doc_embs, logits = [], [], []
        sen_cell_fw = tf.nn.rnn_cell.LSTMCell(
            self.hidden_size // 2,
            forget_bias=0.,
            initializer=self.weights_initializer)
        sen_cell_bw = tf.nn.rnn_cell.LSTMCell(
            self.hidden_size // 2,
            forget_bias=0.,
            initializer=self.weights_initializer)
        for scope, identities, cur_input_x in zip(
            ['user_block', 'product_block'], [[usr], [prd]],
            [nscua_input_x, nscpa_input_x]):
            with tf.variable_scope(scope):
                sen_embs.append(
                    nsc_sentence_layer(
                        cur_input_x,
                        self.max_sen_len,
                        self.max_doc_len,
                        aug_sen_len,
                        identities,
                        self.hidden_size,
                        self.emb_dim,
                        self.sen_hop_cnt,
                        bidirectional_lstm=True,
                        lstm_cells=[sen_cell_fw, sen_cell_bw]))
        lstm_out, _state = lstm(
            tf.reshape(input_x, [-1, self.max_sen_len - 1, self.emb_dim]),
            tf.reshape(sen_len, [-1]),
            self.hidden_size,
            'lstm_bkg',
            bidirectional=True,
            lstm_cells=[sen_cell_fw, sen_cell_bw])
        is_polarity = tf.where(
            tf.equal(polarity, 0), tf.zeros_like(polarity),
            tf.ones_like(polarity))
        is_polarity = tf.to_float(is_polarity)
        is_polarity = tf.reshape(is_polarity, [-1, self.max_sen_len - 1])
        is_polarity_sum = tf.reduce_sum(is_polarity, axis=-1)
        is_polarity_sum = tf.where(
            tf.equal(is_polarity_sum, 0), tf.ones_like(is_polarity_sum),
            is_polarity_sum)
        is_polarity = is_polarity / is_polarity_sum[:, None]
        lstm_out = tf.reshape(lstm_out,
                              [-1, self.max_sen_len - 1, self.hidden_size])
        is_polarity = tf.reshape(is_polarity, [-1, 1, self.max_sen_len - 1])
        lstm_out = tf.matmul(is_polarity, lstm_out)
        lstm_out = tf.reshape(lstm_out,
                              [-1, self.max_doc_len, self.hidden_size])
        sen_embs.append(lstm_out)

        sen_embs = tf.concat(sen_embs, axis=-1)
        sen_embs = tf.layers.dense(
            sen_embs,
            self.emb_dim,
            kernel_initializer=self.weights_initializer,
            bias_initializer=self.biases_initializer)

        nscua_sen_embs, nscpa_sen_embs = sen_embs, sen_embs
        # padding doc with user and product embeddings
        doc_aug_usr = tf.layers.dense(
            usr,
            self.emb_dim,
            use_bias=False,
            kernel_initializer=self.weights_initializer,
            bias_initializer=self.biases_initializer)
        doc_aug_prd = tf.layers.dense(
            prd,
            self.emb_dim,
            use_bias=False,
            kernel_initializer=self.weights_initializer,
            bias_initializer=self.biases_initializer)
        nscua_sen_embs = tf.concat([doc_aug_usr[:, None, :], nscua_sen_embs],
                                   axis=1)
        nscpa_sen_embs = tf.concat([doc_aug_prd[:, None, :], nscpa_sen_embs],
                                   axis=1)
        #  none_sen_embs = tf.pad(sen_embs, [[0, 0], [1, 0], [0, 0]])
        self.max_doc_len += 1
        doc_len = doc_len + 1

        doc_cell_fw = tf.nn.rnn_cell.LSTMCell(
            self.hidden_size // 2,
            forget_bias=0.,
            initializer=self.weights_initializer)
        doc_cell_bw = tf.nn.rnn_cell.LSTMCell(
            self.hidden_size // 2,
            forget_bias=0.,
            initializer=self.weights_initializer)
        for scope, identities, input_x in zip(
            ['user_block', 'product_block'], [[usr], [prd]],
            [nscua_sen_embs, nscpa_sen_embs]):
            with tf.variable_scope(scope):
                doc_emb = nsc_document_layer(
                    input_x,
                    self.max_doc_len,
                    doc_len,
                    identities,
                    self.hidden_size,
                    self.doc_hop_cnt,
                    bidirectional_lstm=True,
                    lstm_cells=[doc_cell_fw, doc_cell_bw])
                doc_embs.append(doc_emb)

                with tf.variable_scope('result'):
                    logits.append(
                        tf.layers.dense(
                            doc_emb,
                            self.cls_cnt,
                            kernel_initializer=self.weights_initializer,
                            bias_initializer=self.biases_initializer))

        nscua_logit, nscpa_logit = logits

        with tf.variable_scope('result'):
            doc_emb = tf.concat(doc_embs, axis=1, name='dhuapa_output')
            logit = tf.layers.dense(
                doc_emb,
                self.cls_cnt,
                kernel_initializer=self.weights_initializer,
                bias_initializer=self.biases_initializer)

        prediction = tf.argmax(logit, 1, name='prediction')

        #  soft_label = tf.nn.softmax(tf.stop_gradient(logit) / (self.cls_cnt / 2))

        with tf.variable_scope("loss"):
            ssce = tf.nn.sparse_softmax_cross_entropy_with_logits
            self.loss = ssce(logits=logit, labels=input_y)
            lossu = ssce(logits=nscua_logit, labels=input_y)
            lossp = ssce(logits=nscpa_logit, labels=input_y)

            self.loss = self.lambda1 * self.loss + self.lambda2 * lossu + self.lambda3 * lossp

            #  sce = tf.nn.softmax_cross_entropy_with_logits_v2
            #  align_lossu = sce(labels=soft_label, logits=nscua_logit)
            #  align_lossp = sce(labels=soft_label, logits=nscpa_logit)
            #  self.loss += .05 * (align_lossu + align_lossp)

            regularizer = tf.zeros(1)
            params = tf.trainable_variables()
            for param in params:
                if param not in self.embeddings.values():
                    regularizer += tf.nn.l2_loss(param)
            self.loss = tf.reduce_mean(self.loss) + self.l2_rate * regularizer

        with tf.variable_scope("metrics"):
            correct_prediction = tf.equal(prediction, input_y)
            mse = tf.reduce_sum(tf.square(prediction - input_y), name="mse")
            correct_num = tf.reduce_sum(
                tf.cast(correct_prediction, dtype=tf.int32),
                name="correct_num")
            accuracy = tf.reduce_sum(
                tf.cast(correct_prediction, "float"), name="accuracy")

        return self.loss, mse, correct_num, accuracy

    def output_metrics(self, metrics, data_length):
        loss, mse, correct_num, _accuracy = metrics
        info = 'Loss = %.3f, RMSE = %.3f, Acc = %.3f' % \
            (loss / data_length, sqrt(float(mse) / data_length), float(correct_num) / data_length)
        return info

    def record_metrics(self, dev_metrics, test_metrics, devlen, testlen):
        _dev_loss, _dev_mse, dev_correct_num, dev_accuracy = dev_metrics
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
        capped_grads_and_vars = []

        for grad, v in grads_and_vars:
            if v is self.embeddings['wrd_emb']:
                grad = tf.IndexedSlices(grad.values * self.embedding_lr,
                                        grad.indices, grad.dense_shape)
            capped_grads_and_vars.append((grad, v))

        train_op = optimizer.apply_gradients(
            capped_grads_and_vars, global_step=global_step)
        return train_op
