from __future__ import unicode_literals
import json
from .lib.dbengine import DBEngine
import re
import numpy as np
from subprocess import Popen, PIPE
from time import sleep
#from nltk.tokenize import StanfordTokenizer
# from ewc import EWC, ewc_train, normal_train, test

schema_re = re.compile(r'\((.+)\)')

def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    max_col_num = 0
    for SQL_PATH in sql_paths:
        print("Loading data from %s" % SQL_PATH)
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                if use_small and idx >= 1000:
                    break
                sql = json.loads(line.strip())
                sql_data.append(sql)

    for TABLE_PATH in table_paths:
        print("Loading data from %s" % TABLE_PATH)
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab

    for sql in sql_data:
        assert sql[u'table_id'] in table_data

    return sql_data, table_data

def load_dataset(dataset_id, use_small=False):
    if dataset_id == 0:
        print("Loading from original dataset")
        sql_data, table_data = load_data('data/train_tok.jsonl',
                'data/train_tok.tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data('data/dev_tok.jsonl',
                'data/dev_tok.tables.jsonl', use_small=use_small)
        test_sql_data, test_table_data = load_data('data/test_tok.jsonl',
                'data/test_tok.tables.jsonl', use_small=use_small)
        TRAIN_DB = 'data/train.db'
        DEV_DB = 'data/dev.db'
        TEST_DB = 'data/test.db'
    else:
        print("Loading from re-split dataset")
        sql_data, table_data = load_data('data_resplit/train.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data('data_resplit/dev.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        test_sql_data, test_table_data = load_data('data_resplit/test.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        TRAIN_DB = 'data_resplit/table.db'
        DEV_DB = 'data_resplit/table.db'
        TEST_DB = 'data_resplit/table.db'

    return sql_data, table_data, val_sql_data, val_table_data,\
            test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB


def load_dataset_dummy(dataset_id, use_small=False, teststr=''):
    if dataset_id == 0:
        print("Loading from original dataset")
        dummy_sql_data, dummy_table_data = load_data('mock/dummy_tok{}.jsonl'.format(teststr), 'mock/dummy_tok.tables.jsonl', use_small=use_small)
    return dummy_sql_data, dummy_table_data


def best_model_name(args, for_load=False, savedstr=''):
    new_data = 'new' if args.dataset > 0 else 'old'
    mode = 'seq2sql' if args.baseline else 'sqlnet'
    if for_load:
        use_emb = use_rl = ''
    else:
        use_emb = '_train_emb' if args.train_emb else ''
        use_rl = 'rl_' if args.rl else ''
    use_ca = '_ca' if args.ca else ''

    agg_model_name = 'saved_model{}/%s_%s%s%s.agg_model'.format(savedstr)%(new_data,
            mode, use_emb, use_ca)
    sel_model_name = 'saved_model{}/%s_%s%s%s.sel_model'.format(savedstr)%(new_data,
            mode, use_emb, use_ca)
    cond_model_name = 'saved_model{}/%s_%s%s%s.cond_%smodel'.format(savedstr)%(new_data,
            mode, use_emb, use_ca, use_rl)

    if not for_load and args.train_emb:
        agg_embed_name = 'saved_model{}/%s_%s%s%s.agg_embed'.format(savedstr)%(new_data,
                mode, use_emb, use_ca)
        sel_embed_name = 'saved_model{}/%s_%s%s%s.sel_embed'.format(savedstr)%(new_data,
                mode, use_emb, use_ca)
        cond_embed_name = 'saved_model{}/%s_%s%s%s.cond_embed'.format(savedstr)%(new_data,
                mode, use_emb, use_ca)

        return agg_model_name, sel_model_name, cond_model_name, agg_embed_name, sel_embed_name, cond_embed_name
    else:
        return agg_model_name, sel_model_name, cond_model_name


def to_batch_seq(sql_data, table_data, idxes, st, ed, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    query_seq = []
    gt_cond_seq = []
    vis_seq = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        q_seq.append(sql['question_tok'])
        col_seq.append(table_data[sql['table_id']]['header_tok'])
        col_num.append(len(table_data[sql['table_id']]['header']))
        ans_seq.append((sql['sql']['agg'],
            sql['sql']['sel'], 
            len(sql['sql']['conds']),
            tuple(x[0] for x in sql['sql']['conds']),
            tuple(x[1] for x in sql['sql']['conds'])))
        query_seq.append(sql['query_tok'])
        gt_cond_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question'],
            table_data[sql['table_id']]['header'], sql['query']))
    if ret_vis_data:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, vis_seq
    else:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq

def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids

def epoch_train(model, optimizer, batch_size, sql_data, table_data, pred_entry):
    model.train()
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        print('batch:', st, 'to', ed, 'of', len(sql_data))

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq = to_batch_seq(sql_data, table_data, perm, st, ed)
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, pred_entry, gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
        loss = model.loss(score, ans_seq, pred_entry, gt_where_seq)
        cum_loss += loss.data.cpu().numpy()[0]*(ed - st)
        optimizer.zero_grad()
        loss = ewc_train(model, optimizer, (q_seq, col_seq), EWC(model), 1000)
        loss.backward()
        optimizer.step()
        st = ed

    return cum_loss / len(sql_data)

def epoch_exec_acc(model, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)

    model.eval()
    perm = list(range(len(sql_data)))
    tot_acc_num = 0.0
    acc_of_log = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num,
                (True, True, True), gt_sel=gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, (True, True, True))

        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid,
                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                        sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None
            tot_acc_num += (ret_gt == ret_pred)
        
        st = ed

    return tot_acc_num / len(sql_data)

# RONNY ADDED
def infer_exec(model, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)
    
    model.eval()
    perm = list(range(len(sql_data)))
    tot_acc_num = 0.0
    acc_of_log = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        
        score = model.forward(q_seq, col_seq, col_num, (True, True, True), gt_sel=gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq, raw_col_seq, (True, True, True))
        for idx, (sql_gt, sql_pred, tid) in enumerate(zip(query_gt, pred_queries, table_ids)):
            print_table(table_data, tid)
            raw_query = engine.get_query_raw(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'], table_data)
            # ret_gt = engine.execute(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            # try:
            #     ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            # except:
            #     ret_pred = None
            # isCorrect = (ret_gt == ret_pred)
            print("==> Here's an example")
            print('English: ', raw_q_seq[0])
            print('SQL: ', raw_query)
            # print 'Execution: ', str(ret_pred[0]).encode('utf-8') if ret_pred else 'null'
            # print 'Correct: ', isCorrect
            print('\n')
            break

        # INFERENCE TIME!
        print('==> Your turn, type a question about this table')
        raw_q_seq, q_seq = input_tokenize_wrapper()
        raw_q_seq = raw_q_seq.decode('utf-8')
        q_seq = [w.decode('utf-8') for w in q_seq]
        raw_q_seq, q_seq = [raw_q_seq, raw_q_seq], [q_seq, q_seq]
        score = model.forward(q_seq, col_seq, col_num, (True, True, True), gt_sel=gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq, raw_col_seq, (True, True, True))
        for idx, (sql_pred, tid) in enumerate(zip(pred_queries, table_ids)):
            raw_query = engine.get_query_raw(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'], table_data, lower=True)
            # try:
            #     ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            # except:
            #     ret_pred = '<Failed>'
            print('ENGLISH: ', raw_q_seq[0])
            print('SQL: ', raw_query)
            # print 'Execution: ', str(ret_pred[0]).encode('utf-8') if ret_pred else 'null'
            print('\n\n')
            break
        st += 1
        sleep(5)
      
# RONNY ADDED
def get_table(table_data, tid):
    table = table_data[tid]
    p = []
    p.append('-' * 80)
    p.append('\n')
    p.append('TABLE {}'.format(tid))
    p.append(table['page_title'])
    p.append(table['section_title'])
    p.append('\t'.join(table['header']).expandtabs(34))
    for row in table['rows']:
        p.append('\t'.join(row).expandtabs(34))
    p.append('\n')
    return '\n'.join(p)

# RONNY ADDED
def input_tokenize_wrapper(english):
    # raw_q = input('English: ')
    raw_q = english
    with open('input_question.txt', 'w') as f:
        f.write(raw_q)
    process = Popen('/opt/conda/bin/python input_tokenize_py3.py ', shell=True, stdout=PIPE, stderr=PIPE)
    process.wait()
    output, err = process.communicate()
    return raw_q, eval(output)

def epoch_acc(model, batch_size, sql_data, table_data, pred_entry, write=False, experiment=None, epoch=0):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    f = open('saved_model/mc.results', 'w')
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, pred_entry, gt_sel = gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq, raw_col_seq, pred_entry)
        one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt, pred_entry)
        
        # formatted = format_preds(raw_q_seq, raw_col_seq, pred_queries)
        # if epoch is not None and not epoch % 10:
        #     for line in formatted.split('\n'):
        #         print(line)
        #         if bool(write):
        #             f.write(line + '\n')
                    
        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)
        st = ed
        
    # if experiment is not None:
    #     experiment.set_step(epoch)
    #     experiment.log_asset(f, file_name='mc_results-' + write)
    f.close()
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)

def make_pred(model, sql_data, table_data, pred_entry=[True, True, True]):
    model.eval()
    perm = list(range(len(sql_data)))
    q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(sql_data, table_data, perm, 0, 2, ret_vis_data=True)
    raw_q_seq = [x[0] for x in raw_data]
    raw_col_seq = [x[1] for x in raw_data]
    query_gt, table_ids = to_batch_query(sql_data, perm, 0, 2)
    gt_sel_seq = [x[1] for x in ans_seq]
    score = model.forward(q_seq, col_seq, col_num, pred_entry, gt_sel = gt_sel_seq)
    pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq, raw_col_seq, pred_entry)
    
    formatted = format_preds(raw_q_seq, raw_col_seq, pred_queries)
    return formatted
    
# RONNY ADDED
def format_preds(raw_q_seq, raw_col_seq, pred_queries):
    
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['EQL', 'GT', 'LT', 'OP']
    
    res= []
    for i in range(len(pred_queries)):
        # initialize output string
        out = ['SELECT']
        
        # select clause
        selcol = raw_col_seq[i][pred_queries[i]['sel']]
        if pred_queries[i]['agg'] != 0:
            sel = agg_ops[pred_queries[i]['agg']] + '(' + selcol + ')'
        else:
            sel = selcol
        out.append(sel)
        
        # from clause
        out.append('FROM mock_time_machine')
        
        # where clause
        out.append('WHERE')
        for j, cond in enumerate(pred_queries[i]['conds']):
            if j != 0:
                out.append('AND')
            wherecol = raw_col_seq[i][cond[0]]
            whereop = cond_ops[cond[1]]
            whereval = cond[2]
            out.append(wherecol)
            out.append(whereop)
            out.append(whereval)
            
        # write to file
        res.append(raw_q_seq[i].replace('\n', '') + ', ' + ' '.join(out) + '\n')

    res.append('\n\n')
    res= ''.join(res)
    return res


def epoch_reinforce_train(model, optimizer, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)

    model.train()
    perm = np.random.permutation(len(sql_data))
    cum_reward = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data =\
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, (True, True, True),
                reinforce=True, gt_sel=gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq,
                raw_col_seq, (True, True, True), reinforce=True)

        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        rewards = []
        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid,
                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                        sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None

            if ret_pred is None:
                rewards.append(-2)
            elif ret_pred != ret_gt:
                rewards.append(-1)
            else:
                rewards.append(1)

        cum_reward += (sum(rewards))
        optimizer.zero_grad()
        model.reinforce_backward(score, rewards)
        optimizer.step()

        st = ed

    return cum_reward / len(sql_data)


def load_word_emb(file_name, load_used=False, use_small=False):
    if not load_used:
        print('Loading word embedding from %s' % file_name)
        ret = {}
        with open(file_name, encoding='utf-8') as inf:
            for idx, line in enumerate(inf):
                if (use_small and idx >= 5000):
                    break
                info = line.strip().split(' ')
                if info[0].lower() not in ret:
                    ret[info[0]] = np.array(list(map(lambda x:float(x), info[1:])))
        return ret
    else:
        print('Load used word embedding')
        with open('glove/word2idx.json') as inf:
            w2i = json.load(inf)
        with open('glove/usedwordemb.npy') as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val
