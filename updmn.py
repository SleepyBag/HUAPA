from functools import partial
from math import sqrt
import tensorflow as tf
from tensorflow import constant as const
from tensorflow.nn import embedding_lookup as lookup
from layers.nsc_sentence_layer import nsc_sentence_layer
from layers.nsc_document_layer import nsc_document_layer
from layers.hop import hop


def var(name, shape, initializer):
    return tf.get_variable(name, shape=shape, initializer=initializer)


class UPDMN(object):
    def __init__(self, args):
        self.max_doc_len = args['max_doc_len']
        self.max_sen_len = args['max_sen_len']
        self.cls_cnt = args['cls_cnt']
        self.embedding = args['embedding']
        self.emb_dim = args['emb_dim']
        self.hidden_size = args['hidden_size']
        self.usr_cnt = args['usr_cnt']
        self.prd_cnt = args['prd_cnt']
        self.doc_cnt = args['doc_cnt']
        self.sen_hop_cnt = args['sen_hop_cnt']
        self.doc_hop_cnt = args['doc_hop_cnt']
        self.l2_rate = args['l2_rate']
        self.convert_flag = ''
        self.debug = args['debug']
        self.lambda1 = args['lambda1']
        self.lambda2 = args['lambda2']
        self.lambda3 = args['lambda3']
        self.embedding_lr = args['embedding_lr']
        self.max_co_doc_cnt = args['max_co_doc_cnt']
        self.hop_cnt = args['hop_cnt']

        self.best_dev_acc = .0
        self.best_test_acc = .0
        self.best_test_rmse = .0

        # initializers for parameters
        self.w_init = tf.contrib.layers.xavier_initializer()
        self.b_init = tf.initializers.zeros()
        self.e_init = tf.contrib.layers.xavier_initializer()

        self.wrd_emb = const(self.embedding, name='wrd_emb', dtype=tf.float32)
        self.doc_mem = var('doc_mem', [self.doc_cnt, self.emb_dim],
                           self.e_init)
        self.usr_emb = var('usr_emb', [self.usr_cnt, self.emb_dim],
                           self.e_init)
        self.prd_emb = var('prd_emb', [self.prd_cnt, self.emb_dim],
                           self.e_init)
        self.embeddings = [
            self.wrd_emb, self.usr_emb, self.prd_emb, self.doc_mem
        ]

        # for tensorboard
        if self.debug:
            tf.summary.histogram('usr_emb', self.usr_emb)
            tf.summary.histogram('prd_emb', self.prd_emb)

    def build(self, data_iter):
        transform = partial(
            tf.layers.dense,
            use_bias=False,
            kernel_initializer=self.w_init,
            bias_initializer=self.b_init)
        dense = partial(
            tf.layers.dense,
            kernel_initializer=self.w_init,
            bias_initializer=self.b_init)
        lstm_cell = partial(
            tf.nn.rnn_cell.LSTMCell,
            self.hidden_size // 2,
            forget_bias=0.,
            initializer=self.w_init)

        def pad_context(context, input_x):
            """ padding content with context embedding """
            tiled_context = transform(context, self.emb_dim)
            tiled_context = tf.tile(tiled_context[:, None, None, :],
                                    [1, self.max_doc_len, 1, 1])
            input_x = tf.reshape(
                input_x,
                [-1, self.max_doc_len, self.max_sen_len, self.emb_dim])
            input_x = tf.concat([tiled_context, input_x], axis=2)
            input_x = tf.reshape(input_x,
                                 [-1, self.max_sen_len + 1, self.emb_dim])
            return input_x

        # get the inputs
        with tf.variable_scope('inputs'):
            input_map = data_iter.get_next()
            usrid, prdid, docid, input_x, input_y, sen_len, doc_len, co_doc, co_doc_cnt = \
                (input_map['usr'], input_map['prd'], input_map['docid'],
                 input_map['content'], input_map['rating'],
                 input_map['sen_len'], input_map['doc_len'],
                 input_map['co_doc'], input_map['co_doc_cnt'])

            usr = lookup(self.usr_emb, usrid)
            prd = lookup(self.prd_emb, prdid)
            usr = tf.stop_gradient(usr)
            prd = tf.stop_gradient(prd)
            input_x = lookup(self.wrd_emb, input_x)
            co_doc = tf.reshape(co_doc, [-1, self.max_co_doc_cnt])

        nscua_input_x = pad_context(usr, input_x)
        nscpa_input_x = pad_context(prd, input_x)

        sen_len = tf.where(
            tf.equal(sen_len, 0), tf.zeros_like(sen_len), sen_len + 1)
        self.max_sen_len += 1

        # build the process of model
        sen_embs, doc_embs = [], []
        sen_cell_fw = lstm_cell()
        sen_cell_bw = lstm_cell()
        for scope, identities, input_x, attention_type in zip(
                ['user_block', 'product_block'], [[usr], [prd]],
                [nscua_input_x, nscpa_input_x], ['additive', 'additive']):
            with tf.variable_scope(scope):
                sen_emb = nsc_sentence_layer(
                    input_x,
                    self.max_sen_len,
                    self.max_doc_len,
                    sen_len,
                    identities,
                    self.hidden_size,
                    self.emb_dim,
                    self.sen_hop_cnt,
                    bidirectional_lstm=True,
                    lstm_cells=[sen_cell_fw, sen_cell_bw],
                    auged=True,
                    attention_type=attention_type)
                sen_embs.append(sen_emb)

        sen_embs = tf.concat(sen_embs, axis=-1)

        # padding doc with user and product embeddings
        doc_aug_usr = transform(usr, 2 * self.hidden_size)
        nscua_sen_embs = tf.concat([doc_aug_usr[:, None, :], sen_embs], axis=1)
        doc_aug_prd = transform(prd, 2 * self.hidden_size)
        nscpa_sen_embs = tf.concat([doc_aug_prd[:, None, :], sen_embs], axis=1)
        #  none_sen_embs = tf.pad(sen_embs, [[0, 0], [1, 0], [0, 0]])
        self.max_doc_len += 1
        doc_len = doc_len + 1

        doc_cell_fw = lstm_cell()
        doc_cell_bw = lstm_cell()
        for scope, identities, input_x, attention_type in zip(
                ['user_block', 'product_block'], [[usr], [prd]],
                [nscua_sen_embs, nscpa_sen_embs], ['additive', 'additive']):
            with tf.variable_scope(scope):
                doc_emb = nsc_document_layer(
                    input_x,
                    self.max_doc_len,
                    doc_len,
                    identities,
                    self.hidden_size,
                    self.doc_hop_cnt,
                    bidirectional_lstm=True,
                    lstm_cells=[doc_cell_fw, doc_cell_bw],
                    auged=True,
                    attention_type=attention_type)
                doc_embs.append(doc_emb)
        doc_emb = tf.concat(doc_embs, axis=1, name='dhuapa_output')
        doc_emb = tf.stop_gradient(doc_emb)

        stopped_doc_mem = tf.stop_gradient(self.doc_mem)
        upd = lookup(stopped_doc_mem, co_doc)
        for _ in range(self.hop_cnt):
            doc_emb = hop('memory_network', upd, upd, doc_emb, None,
                          co_doc_cnt, self.max_co_doc_cnt, 'o', attention_type='additive')
        #  tf.scatter_update(self.doc_mem, docid, doc_emb)

        with tf.variable_scope('result'):
            logit = dense(doc_emb, self.cls_cnt)

        prediction = tf.argmax(logit, 1, name='prediction')

        #  soft_label = tf.nn.softmax(tf.stop_gradient(logit) / (self.cls_cnt / 2))

        with tf.variable_scope("loss"):
            ssce = tf.nn.sparse_softmax_cross_entropy_with_logits
            self.loss = ssce(logits=logit, labels=input_y)

            regularizer = tf.zeros(1)
            params = tf.trainable_variables()
            for param in params:
                if param not in self.embeddings:
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
        info = 'Loss = %.5f, RMSE = %.3f, Acc = %.3f' % \
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
            if v is self.wrd_emb:
                grad = tf.IndexedSlices(grad.values * self.embedding_lr,
                                        grad.indices, grad.dense_shape)
            capped_grads_and_vars.append((grad, v))

        train_op = optimizer.apply_gradients(
            capped_grads_and_vars, global_step=global_step)
        return train_op
