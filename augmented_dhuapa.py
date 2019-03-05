from math import sqrt
import tensorflow as tf
from tensorflow import constant as const
from layers.nsc_sentence_layer import nsc_sentence_layer
from layers.nsc_document_layer import nsc_document_layer
lookup = tf.nn.embedding_lookup


def var(name, shape, initializer):
    return tf.get_variable(name, shape=shape, initializer=initializer)


class DHUAPA_MUTUAL(object):
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

        hsize = self.hidden_size

        # embeddings in the model
        with tf.variable_scope('emb'):
            self.embeddings = {
                'wrd_emb':
                const(self.embedding, name='wrd_emb', dtype=tf.float32),
                #  'wrd_emb': tf.Variable(self.embedding, name='wrd_emb', dtype=tf.float32),
                'usr_emb':
                var('usr_emb', [self.usr_cnt, self.emb_dim], self.emb_initializer),
                'prd_emb':
                var('prd_emb', [self.prd_cnt, self.emb_dim], self.emb_initializer)
            }

        # for tensorboard
        if self.debug:
            tf.summary.histogram('usr_emb', self.embeddings['usr_emb'])
            tf.summary.histogram('prd_emb', self.embeddings['prd_emb'])

    def build(self, data_iter):
        # get the inputs
        with tf.variable_scope('inputs'):
            input_map = data_iter.get_next()
            usrid, prdid, input_x, input_y, sen_len, doc_len = \
                (input_map['usr'], input_map['prd'],
                 input_map['content'], input_map['rating'],
                 input_map['sen_len'], input_map['doc_len'])

            usr = lookup(
                self.embeddings['usr_emb'], usrid, name='cur_usr_embedding')
            prd = lookup(
                self.embeddings['prd_emb'], prdid, name='cur_prd_embedding')
            input_x = lookup(
                self.embeddings['wrd_emb'], input_x, name='cur_wrd_embedding')

            nscua_input_x, nscpa_input_x = input_x, input_x
            # padding content with user embedding
            nscua_input_x = tf.reshape(
                input_x,
                [-1, self.max_doc_len, self.max_sen_len, self.emb_dim])
            nscua_input_x = tf.concat([
                tf.tile(usr[:, None, None, :], [1, self.max_doc_len, 1, 1]),
                nscua_input_x
            ], axis=2)
            nscpa_input_x = tf.reshape(
                input_x,
                [-1, self.max_doc_len, self.max_sen_len, self.emb_dim])
            nscpa_input_x = tf.concat([
                tf.tile(prd[:, None, None, :], [1, self.max_doc_len, 1, 1]),
                nscpa_input_x
            ], axis=2)
            sen_len = sen_len + 1
            self.max_sen_len += 1
            nscua_input_x = tf.reshape(nscua_input_x,
                                       [-1, self.max_sen_len, self.emb_dim])
            nscpa_input_x = tf.reshape(nscpa_input_x,
                                       [-1, self.max_sen_len, self.emb_dim])

        # build the process of model
        doc_embs, logits = [], []
        for scope, identities, input_x in zip(['user_block', 'product_block'],
                                              [[usr], [prd]],
                                              [nscua_input_x, nscpa_input_x]):
            with tf.variable_scope(scope):
                doc_emb = nsc(input_x, self.max_sen_len, self.max_doc_len,
                              sen_len, doc_len, identities, self.hidden_size,
                              self.emb_dim, self.sen_hop_cnt, self.doc_hop_cnt)
                doc_embs.append(doc_emb)

                with tf.variable_scope('result'):
                    logits.append(
                        tf.layers.dense(
                            doc_emb,
                            self.cls_cnt,
                            kernel_initializer=self.weights_initializer,
                            bias_initializer=self.biases_initializer))

        nscua_logit, nscpa_logit = logits
        nscua_soft_label = tf.nn.softmax(
            tf.stop_gradient(nscua_logit) / (self.cls_cnt / 2))
        nscpa_soft_label = tf.nn.softmax(
            tf.stop_gradient(nscpa_logit) / (self.cls_cnt / 2))
        tf.summary.histogram('nscua_soft_label', nscua_soft_label)
        tf.summary.histogram('nscpa_soft_label', nscpa_soft_label)

        with tf.variable_scope('result'):
            doc_emb = tf.concat(doc_embs, axis=1, name='dhuapa_output')
            logit = tf.layers.dense(
                doc_emb,
                self.cls_cnt,
                kernel_initializer=self.weights_initializer,
                bias_initializer=self.biases_initializer)

        prediction = tf.argmax(logit, 1, name='prediction')

        with tf.variable_scope("loss"):
            ssce = tf.nn.sparse_softmax_cross_entropy_with_logits
            self.loss = ssce(logits=logit, labels=input_y)
            lossu = ssce(logits=nscua_logit, labels=input_y)
            lossp = ssce(logits=nscpa_logit, labels=input_y)

            self.loss = self.lambda1 * self.loss + self.lambda2 * lossu + self.lambda3 * lossp

            sce = tf.nn.softmax_cross_entropy_with_logits_v2
            align_lossu = sce(labels=nscpa_soft_label, logits=nscua_logit)
            align_lossp = sce(labels=nscua_soft_label, logits=nscpa_logit)
            #  self.loss += 0. * (align_lossu + align_lossp)

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
            #  if var is self.embeddings['wrd_emb'] or var is self.embeddings['usr_emb']:
            #      grad = tf.IndexedSlices(grad.values * self.embedding_lr,
            #                              grad.indices, grad.dense_shape)
            capped_grads_and_vars.append((grad, v))

        train_op = optimizer.apply_gradients(
            capped_grads_and_vars, global_step=global_step)
        return train_op
