from __future__ import division
from comet_ml import Experiment
import json
import torch
from sqlnet.utils import *
from sqlnet.model.seq2sql import Seq2SQL
from sqlnet.model.sqlnet import SQLNet
import numpy as np
import datetime
import copy
import pdb
import argparse
import os
from time import time
from flask import Flask, request
import sys


app = Flask(__name__) 

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='2,3')
parser.add_argument('--toy', action='store_true',
        help='If set, use small data; used for fast debugging.')
parser.add_argument('--suffix', type=str, default='',
        help='The suffix at the end of saved model name.')
parser.add_argument('--ca', action='store_true',
        help='Use conditional attention.')
parser.add_argument('--dataset', type=int, default=0,
        help='0: original dataset, 1: re-split dataset')
parser.add_argument('--rl', action='store_true',
        help='Use RL for Seq2SQL(requires pretrained model).')
parser.add_argument('--baseline', action='store_true',
        help='If set, then train Seq2SQL model; default is SQLNet model.')
parser.add_argument('--train_emb', action='store_true',
        help='Train word embedding for SQLNet(requires pretrained model).')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

N_word=300
B_word=42
# B_word=6
if args.toy:
    USE_SMALL=True
    GPU=True
    BATCH_SIZE=15
else:
    USE_SMALL=False
    GPU=True
    BATCH_SIZE=64
TRAIN_ENTRY=(True, True, True)  # (AGG, SEL, COND)
TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
learning_rate = 1e-4 if args.rl else 1e-3

# load wikisql
sql_data, table_data, val_sql_data, val_table_data, test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB = load_dataset(args.dataset, use_small=USE_SMALL)
pdb.set_trace()
# concatenate mastercard dummy dataset with wikisql
dummy_sql_data, dummy_table_data = load_dataset_dummy(0)
[sql_data.extend(dummy_sql_data) for _ in range(100)]
table_data.update(dummy_table_data)

# see train results on mc data
dum_sql_data = dummy_sql_data
dum_table_data = dummy_table_data

# see test results on mc data
dummy_sql_data, dummy_table_data = load_dataset_dummy(0, teststr='_test')
val_sql_data = dummy_sql_data
val_table_data = dummy_table_data

# load word embedding
tic = time()
print('==> loading word embedding')
word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), load_used=args.train_emb, use_small=USE_SMALL)

#words = ['what', 'are', 'the', 'account', 'numbers', 'with', 'open', 'account', 'days', 'below', '120', 'days', '?']
#selected_emb = {}
#for i, key in enumerate(words):
#    selected_emb[key] = word_emb[key]
#import pickle
#print('selected_emb', selected_emb)
#f = open("glove/selected_emb.pkl","wb")
#pickle.dump(selected_emb,f)
#f.close()

#f2 = open("glove/selected_emb.pkl", 'rb')
#word_emb = pickle.load(f2)
#f2.close()

# with open('glove/word_emb.&dB.%dd_xc.pkl' % (B_word, N_word), encoding='utf-8') as f:
#     # pickle.load(f)
#     word_emb = pickle.load(f, encoding='utf-8')
print('time to load word emb: ' + str(time() - tic))

# build sqlnet model
if not args.baseline:
    tic = time()
    print('==> loading sqlnet constructor')
    model = SQLNet(word_emb, N_word=N_word, use_ca=args.ca, gpu=GPU, trainable_emb = args.train_emb)
    print('time to load sqlnet constructor: ' + str(time() - tic))
    assert not args.rl, "SQLNet can\'t do reinforcement learning."
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0)

if args.train_emb:
    agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(args)

# load model
agg_m, sel_m, cond_m = best_model_name(args, savedstr='_pretrain_wikisql')
print('==> best model names:', agg_m, sel_m, cond_m)
print("Loading from %s" % agg_m)
model.agg_pred.load_state_dict(torch.load(agg_m))
print("Loading from %s" % sel_m)
model.sel_pred.load_state_dict(torch.load(sel_m))
print("Loading from %s" % cond_m)
model.cond_pred.load_state_dict(torch.load(cond_m))
if args.rl or args.train_emb:
    print('train_emb is on, so loading best_model')
    agg_lm, sel_lm, cond_lm = best_model_name(args, for_load=True)
    print("Loading from %s" % agg_lm)
    model.agg_pred.load_state_dict(torch.load(agg_lm))
    print("Loading from %s" % sel_lm)
    model.sel_pred.load_state_dict(torch.load(sel_lm))
    print("Loading from %s" % cond_lm)
    model.cond_pred.load_state_dict(torch.load(cond_lm))

# function to run inference
def inference(english):
    # pdb.set_trace()
    infer_data = copy.deepcopy(dum_sql_data[:2])
    raw_q_seq, q_seq = input_tokenize_wrapper(english)
    # raw_q_seq = raw_q_seq.decode('utf-8')
    infer_data[0]['question'] = raw_q_seq
    infer_data[0]['question_tok'] = q_seq
    formatted = make_pred(model, infer_data, table_data)
    return formatted.split('\n')[0].split(',')[-1]

# get dummy table
tablestr = get_table(dummy_table_data, 'mock_time_machine')
print(tablestr)

test_question = "what are the account numbers with open account days below 120 days?"
# USE THIS FOR DEBUGGING
print(test_question)
# print(inference(test_question))

question_list = ["how many accounts whose gender is female and spend less than 100 in food",
                "tell me account number that spend more than 100 in information technology and with age under 30",
                "how many distinct accounts",
                "what is minimum age of users?",
                "what is the average food spending of users who are between the age of 20 and 30", 
                "present the users who spent less than 100 in food and whose gender is female",
                "how many users who spend less than 100 in food and whose gender is female"]


# for i, q in enumerate(question_list):
#    print('i', i, 'q:', q)
#    print("sql:", inference(q))

# start flask app
# if running in docker, must also create localhost tunnel by running the following from the home folder or wherever pagekite.py is
# python2 pagekite.py 5000 wronnyhuang.pagekite.me
# then you can do get, i.e., the table by going to https://wronnyhuang.pagekite.me/table
@app.route('/', methods=['GET'])
def get_sql():
    english = request.args.get('english', default='', type=str)
    return inference(english)
    # return english

@app.route('/table')
def get_table():
    return tablestr

if __name__ == '__main__':
   # app.run()
    # print(inference("what are the account numbers with open account days below 120 days?"))
    
    test_question = "what are the account numbers with open account days below 120 days?"
    # print('test_question:', test_question)
    # print(inference(test_question))
    print('python version: \n', sys.version)
    try:
        while True:
            question = input('what is your question?   \n')
            print('SQL query: \n') 
            print(inference(question))
            print('\n')
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print('Exception: ', str(e))

            
