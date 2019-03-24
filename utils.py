from tqdm import tqdm
import math
import data
from tensorflow.python import pywrap_tensorflow


def run_set(sess, step_cnt, sealed_metrics, ops=[]):
    sealed_metrics = [(a, 'SUM') if not isinstance(a, tuple) else a
                      for a in sealed_metrics]
    metrics = sealed_metrics + ops
    pgb = tqdm(range(step_cnt), leave=False, dynamic_ncols=True)
    ans = []
    for metric, method in metrics:
        if method == 'ALL':
            ans.append([])
        elif method == 'MEAN' or method == 'SUM':
            ans.append(0)
        elif method == 'LAST' or method == 'NONE':
            ans.append(None)
    methods = [metric[1] for metric in metrics]
    metrics = [metric[0] for metric in metrics]
    for _ in pgb:
        cur_metrics = sess.run(metrics)
        for i, metric, method in zip(range(len(ans)), cur_metrics, methods):
            if method == 'ALL':
                ans[i].append(metric)
            elif method == 'MEAN':
                ans[i] += metric / step_cnt
            elif method == 'SUM':
                ans[i] += metric
            elif method == 'LAST':
                ans[i] = metric
            elif method == 'NONE':
                pass
    return [ans[:len(sealed_metrics)]] + ans[len(sealed_metrics):]


def load_data(dataset, drop, emb_dim, batch_size, user_pretrain, max_doc_len,
              max_sen_len, repeat, split_by_period):
    # Load data
    print("Loading data...")
    datasets = [
        'data/' + dataset + s for s in ['/train.ss', '/dev.ss', '/test.ss']
    ]
    tfrecords = [
        'data/' + dataset + '/tfrecords' + s
        for s in ['/train.tfrecord', '/dev.tfrecord', '/test.tfrecord']
    ]
    stats_filename = 'data/' + dataset + '/stats/stats.txt' + str(drop)
    embedding_filename = 'data/' + dataset + '/embedding/embedding' + str(
        emb_dim) + ('user' if user_pretrain else '') + '.txt'
    print(embedding_filename)
    user_embedding_filename = 'data/' + dataset + '/embedding/user_embedding.txt' if user_pretrain else ''
    text_filename = 'data/' + dataset + '/word2vec_train.ss'
    datasets, lengths, embedding, user_embedding, stats, wrd_dict = data.build_dataset(
        datasets, tfrecords, stats_filename, embedding_filename,
        user_embedding_filename, max_doc_len, max_sen_len, split_by_period,
        emb_dim, text_filename, drop)
    trainset, devset, testset = datasets
    trainlen, devlen, testlen = lengths
    #  trainlen *=  1 - flags.drop
    if repeat:
        trainset = trainset.repeat()
        devset = devset.repeat()
        testset = testset.repeat()
    trainset = trainset.shuffle(300000).batch(batch_size).repeat()
    devset = devset.shuffle(300000).batch(batch_size)
    testset = testset.shuffle(300000).batch(batch_size)
    print("Data loaded.")
    return embedding, user_embedding, trainset, devset, testset, trainlen, devlen, testlen, stats


def get_step_cnt(datalen, batch_size):
    return int(math.ceil(float(datalen) / batch_size))


def get_variables_to_restore(variables, var_keep_dic):
    variables_to_restore = []
    for v in variables:
        # one can do include or exclude operations here.
        if v.name.split(':')[0] in var_keep_dic:
            print("Variables restored: %s" % v.name)
            variables_to_restore.append(v)

    return variables_to_restore

def get_variable_in_checkpoint_file(checkpoint_path):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return list(var_to_shape_map.keys())

