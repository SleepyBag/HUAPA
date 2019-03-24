# -*- coding: utf-8 -*-
# author: Xue Qianming
import os
import time
import pickle
import numpy as np
import tensorflow as tf
import data
from colored import fg, stylize
import math
from tensorflow.python import debug as tf_debug
from utils import run_set, load_data, get_step_cnt

# delete all flags that remained by last run
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

try:
    del_all_flags(tf.flags.FLAGS)
except:
    pass

params = {
    'debug_params': [('debug', False, 'Whether to debug or not'),
                     ('tfdbg', 'null', 'the manner to debug(cli or tensorboard)'),
                     ('check', False, 'Whether to make a checkpoint')],
    'data_params': [('cls_cnt', 10, "Numbers of class"),
                    ('drop', .0, "How much percent of data will be dropped"),
                    ('user_pretrain', False,
                     "whether to pretrain user embedding or not"),
                    ('dataset', 'test', "The dataset")],
    'model_chooing': [('model', 'dhuapa', 'Model to train')],
    'model_hyperparam':
    [("emb_dim", 100, "Dimensionality of character embedding"),
     ("hidden_size", 100, "hidden_size"),
     ('max_sen_len', 50, 'max number of tokens per sentence'),
     ('max_doc_len', 40, 'max number of sentences per document'),
     ('max_co_doc_cnt', 1000, 'max number of the sum of U(d) and P(d)'),
     ('sen_aspect_cnt', 1, 'max number of tokens per sentence'),
     ('doc_aspect_cnt', 1, 'max number of tokens per document'),
     ('sen_hop_cnt', 1, 'layers of memnet in sentence layer'),
     ('doc_hop_cnt', 1, 'layers of memnet in document layer'),
     ('hop_cnt', 1, 'layers of memnet in the end'),
     ("lr", .005, "Learning rate"), ("l2_rate", 0.,
                                     "rate of l2 regularization"),
     ("embedding_lr", 1e-5, "embedding learning rate"),
     ("lambda1", .4, "proportion of the total loss"),
     ("lambda2", .3, "proportion of the loss of user block"),
     ("lambda3", .3, "proportion of the loss of product block"),
     ("bilstm", True, "use biLSTM or LSTM"),
     ("split_by_period", True,
      "whether to split the document by sentences or fixed length")],
    'training_params': [("batch_size", 100, "Batch Size"),
                        ("epoch_cnt", 10, "Number of training epochs"),
                        ("checkpoint", '', "checkpoint to restore params"),
                        ("training_method", 'adam',
                         'Method chose to tune the weights')],
    'misc_params':
    [("allow_soft_placement", True, "Allow device soft device placement"),
     ("log_device_placement", False, "Log placement of ops on devices")]
}

for param_collection in list(params.values()):
    for param_name, default, description in param_collection:
        param_type = type(default)
        if param_type is int:
            tf.flags.DEFINE_integer(param_name, default, description)
        elif param_type is float:
            tf.flags.DEFINE_float(param_name, default, description)
        elif param_type is str:
            tf.flags.DEFINE_string(param_name, default, description)
        elif param_type is bool:
            tf.flags.DEFINE_boolean(param_name, default, description)

flags = tf.flags.FLAGS

# save current codes
cur_time = time.time()
os.system('mkdir code_history/' + str(cur_time))
os.system('cp *.py code_history/' + str(cur_time) + '/')
localtime = time.localtime(cur_time)

_ = flags.batch_size
# print params
output_file = open('code_history/' + str(cur_time) + '/output.txt', 'a')
print("\nParameters:")
for attr, value in sorted(flags.__flags.items()):
    print(("{}={}".format(attr.upper(), value.value)))
    print("{}={}".format(attr.upper(), value.value), file=output_file)
print("")
output_file.close()

embedding, user_embedding, trainset, devset, testset, trainlen, devlen, testlen, stats = load_data(
    flags.dataset,
    flags.drop,
    flags.emb_dim,
    flags.batch_size,
    flags.user_pretrain,
    flags.max_doc_len,
    flags.max_sen_len,
    repeat=False,
    split_by_period=flags.split_by_period)
usr_cnt, prd_cnt, doc_cnt = stats['usr_cnt'], stats['prd_cnt'], stats['doc_cnt']
if 'alternate' in flags.model:
    _, _, full_trainset, full_devset, full_testset, _, _, _, _, _ = load_data(
        flags.dataset,
        .0,
        flags.emb_dim,
        flags.batch_size,
        flags.user_pretrain,
        flags.max_doc_len,
        flags.max_sen_len,
        repeat=True,
        split_by_period=flags.split_by_period)


# create the session
session_config = tf.ConfigProto(
    allow_soft_placement=flags.allow_soft_placement,
    log_device_placement=flags.log_device_placement
)
session_config.gpu_options.allow_growth = True
sess = tf.Session(config=session_config)

if flags.tfdbg == 'cli':
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

# build the model
model_params = {
    'max_sen_len': flags.max_sen_len,
    'max_doc_len': flags.max_doc_len,
    'max_co_doc_cnt': flags.max_co_doc_cnt,
    'sen_aspect_cnt': flags.sen_aspect_cnt,
    'doc_aspect_cnt': flags.doc_aspect_cnt,
    'cls_cnt': flags.cls_cnt,
    'embedding': embedding,
    'emb_dim': flags.emb_dim,
    'hidden_size': flags.hidden_size,
    'usr_cnt': usr_cnt,
    'prd_cnt': prd_cnt,
    'doc_cnt': doc_cnt,
    'l2_rate': flags.l2_rate,
    'debug': flags.debug,
    'lambda1': flags.lambda1,
    'lambda2': flags.lambda2,
    'lambda3': flags.lambda3,
    'sen_hop_cnt': flags.sen_hop_cnt,
    'doc_hop_cnt': flags.doc_hop_cnt,
    'hop_cnt': flags.hop_cnt,
    'embedding_lr': flags.embedding_lr
}
exec('from ' + flags.model + ' import ' + flags.model.upper() + ' as model')
model = model(model_params)

data_iter = tf.data.Iterator.from_structure(trainset.output_types,
                                            output_shapes=trainset.output_shapes)
traininit = data_iter.make_initializer(trainset)
devinit = data_iter.make_initializer(devset)
testinit = data_iter.make_initializer(testset)

metrics = model.build(data_iter)

# restore the params
saver = tf.train.Saver()
saver.restore(sess, 'ckpts/' + flags.model + '/-' + flags.checkpoint)

wrd = 'divide0/fold/cur_wrd:0'
usr_attention = []
prd_attention = []

global_step = tf.Variable(0, name="global_step", trainable=False)
# run a dataset

try:
    for epoch in range(flags.epoch_cnt):
        sess.run(traininit)
        output_file = open('code_history/' + str(cur_time) + '/output.txt', 'a')
        ops = [(model.doc_mem, 'LAST')]
        # train on trainset
        train_metrics, train_doc_mem = run_set(sess, get_step_cnt(trainlen, flags.batch_size), metrics, ops)
        info = model.output_metrics(train_metrics, trainlen)
        info = 'Trainset:' + info
        print(stylize(info, fg('yellow')))
        import ipdb; ipdb.set_trace()
        np.save('data/' + flags.dataset + '/doc_mem/' + flags.model + '_train', train_doc_mem)

        # test on devset
        sess.run(devinit)
        dev_metrics, dev_doc_mem = run_set(sess, get_step_cnt(devlen, flags.batch_size), metrics, ops)
        info = model.output_metrics(dev_metrics, devlen)
        info = 'Devset:  ' + info
        print(stylize(info, fg('green')))
        np.save('data/' + flags.dataset + '/doc_mem/' + flags.model + '_dev', dev_doc_mem)

        # test on testset
        sess.run(testinit)
        test_metrics, test_doc_mem = run_set(sess, get_step_cnt(testlen, flags.batch_size), metrics, ops)
        info = model.output_metrics(test_metrics, testlen)
        info = 'Testset: ' + info
        print(stylize(info, fg('red')))
        np.save('data/' + flags.dataset + '/doc_mem/' + flags.model + '_test', test_doc_mem)

        # print info of this epoch
        info = model.record_metrics(dev_metrics, test_metrics, devlen, testlen)
        info = 'Epoch %d finished, ' % epoch + info
        print(stylize(info, fg('white')))

except KeyboardInterrupt:
    print('Interrupted')
    best_test_acc = model.best_test_acc
    src = 'code_history/' + str(cur_time)
    dest = 'code_history/' + 'acc' + str(best_test_acc) + '_' + str(cur_time)
    os.system('mv ' + src + ' ' + dest)
