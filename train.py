# -*- coding: utf-8 -*-
# author: Xue Qianming
import os
import time
import tensorflow as tf
from colored import fg, stylize
from utils import run_set, load_data, get_step_cnt, get_variable_in_checkpoint_file, get_variables_to_restore

params = {
    'debug_params': [('debug', False, 'Whether to debug or not'),
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
usr_cnt, prd_cnt, doc_cnt = stats['usr_cnt'], stats['prd_cnt'], stats[
    'doc_cnt']
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

# save current codes
cur_time = time.time()
os.system('mkdir code_history/' + str(cur_time))
os.system('cp *.py code_history/' + str(cur_time) + '/')
localtime = time.localtime(cur_time)

# force to parse flags
_ = flags.batch_size
# print params
output_file = open('code_history/' + str(cur_time) + '/output.txt', 'a')
print("\nParameters:")
for attr, value in sorted(flags.__flags.items()):
    print(("{}={}".format(attr.upper(), value.value)))
    print("{}={}".format(attr.upper(), value.value), file=output_file)
print("")
output_file.close()

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
os.system("figlet -f slant " + flags.model)

# create the session
session_config = tf.ConfigProto(
    allow_soft_placement=flags.allow_soft_placement,
    log_device_placement=flags.log_device_placement)
session_config.gpu_options.allow_growth = True
sess = tf.Session(config=session_config)

# create data iterators
data_iter = tf.data.Iterator.from_structure(
    trainset.output_types, output_shapes=trainset.output_shapes)
traininit = data_iter.make_initializer(trainset)
devinit = data_iter.make_initializer(devset)
testinit = data_iter.make_initializer(testset)
if 'alternate' in flags.model:
    full_data_iter = tf.data.Iterator.from_structure(
        full_trainset.output_types, output_shapes=full_trainset.output_shapes)
    full_traininit = full_data_iter.make_initializer(full_trainset)
    full_devinit = full_data_iter.make_initializer(full_devset)
    full_testinit = full_data_iter.make_initializer(full_testset)
    sess.run(full_traininit)
    sess.run(full_devinit)
    sess.run(full_testinit)

# build the model graph
if 'alternate' in flags.model:
    metrics = model.build(data_iter, full_data_iter)
else:
    metrics = model.build(data_iter)

# Define Training procedure
global_step = tf.Variable(0, name="global_step", trainable=False)
if flags.training_method == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(flags.lr)
elif flags.training_method == 'adam':
    optimizer = tf.train.AdamOptimizer(flags.lr)
elif flags.training_method == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(flags.lr, epsilon=1e-6)
train_op = model.train(optimizer, global_step)

# merge tensorboard summary
summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('summary/train', sess.graph)
# dev_writer = tf.summary.FileWriter('summary/dev', sess.graph)
# test_writer = tf.summary.FileWriter('summary/test', sess.graph)

if flags.checkpoint == '':
    sess.run(tf.global_variables_initializer())
else:
    # restore the params
    checkpoint_path = os.path.join(
        'ckpts', flags.model,
        flags.checkpoint) if '/' not in flags.checkpoint else os.path.join(
            'ckpts', flags.checkpoint)
    global_variables = tf.global_variables()
    var_keep_dic = get_variable_in_checkpoint_file(checkpoint_path)
    variable_to_restore = get_variables_to_restore(global_variables,
                                                   var_keep_dic)
    saver = tf.train.Saver(variable_to_restore)
    saver.restore(sess, 'ckpts/' + flags.checkpoint)

    # initialize other params
    uninitialized_vars = sess.run(tf.report_uninitialized_variables())
    uninitialized_vars = [s.decode() + ':0' for s in uninitialized_vars]
    uninitialized_vars = [
        var for var in tf.global_variables() if var.name in uninitialized_vars
    ]
    init_uninitialized_op = tf.initialize_variables(uninitialized_vars)
    sess.run(init_uninitialized_op)

if flags.check:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

try:
    for epoch in range(flags.epoch_cnt):
        sess.run(traininit)
        output_file = open('code_history/' + str(cur_time) + '/output.txt',
                           'a')
        # train on trainset
        # trainlen = flags.batch_size * flags.evaluate_every
        # when debugging, summary info is needed for tensorboard
        # cur_trainlen = trainlen if model.best_test_acc < 0.530 \
        #     else flags.evaluate_every * flags.batch_size
        if summary is not None:
            train_metrics, step, train_summary, _ = run_set(
                sess, get_step_cnt(trainlen, flags.batch_size), metrics,
                [(global_step, 'ALL'), (summary, 'ALL'), (train_op, 'NONE')])
        else:
            train_metrics, step, _ = run_set(
                sess, get_step_cnt(trainlen, flags.batch_size), metrics,
                [(global_step, 'ALL'), (train_op, 'NONE')])
        #  train_metrics, step, _ = \
        #      run_set(sess, trainlen, metrics, (global_step, train_op, ))
        info = model.output_metrics(train_metrics, trainlen)
        info = 'Trainset:' + info
        print((stylize(info, fg('yellow'))))
        print(info, file=output_file)

        if summary is not None:
            for i, s in zip(step, train_summary):
                train_writer.add_summary(s, i)
                train_writer.flush()

        # test on devset
        sess.run(devinit)
        dev_metrics, = run_set(sess, get_step_cnt(devlen, flags.batch_size),
                               metrics)
        info = model.output_metrics(dev_metrics, devlen)
        info = 'Devset:  ' + info
        print((stylize(info, fg('green'))))
        print(info, file=output_file)

        # test on testset
        sess.run(testinit)
        test_metrics, = run_set(sess, get_step_cnt(testlen, flags.batch_size),
                                metrics)
        info = model.output_metrics(test_metrics, testlen)
        info = 'Testset: ' + info
        print((stylize(info, fg('red'))))
        print(info, file=output_file)

        # print info of this epoch
        info = model.record_metrics(dev_metrics, test_metrics, devlen, testlen)
        info = 'Epoch %d finished, ' % epoch + info
        print((stylize(info, fg('white'))))
        print(info, file=output_file)

        # write a checkpoint
        if flags.check and 'NEW' in info:
            try:
                os.mkdir('ckpts/' + flags.model)
            except:
                pass
            save_path = saver.save(
                sess, 'ckpts/' + flags.model + '/', global_step=step[-1])
            print(('Checkpoint saved to ' + save_path))

        output_file.close()

except KeyboardInterrupt:
    print('Interrupted')
    best_test_acc = model.best_test_acc
    src = 'code_history/' + str(cur_time)
    dest = 'code_history/' + 'acc' + str(best_test_acc) + '_' + str(cur_time)
    os.system('mv ' + src + ' ' + dest)
