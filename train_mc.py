from comet_ml import Experiment
experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False, project_name='sqlnet', workspace="wronnyhuang")
experiment.set_name('train_mc')

import json
import torch
from sqlnet.utils import *
from sqlnet.model.seq2sql import Seq2SQL
from sqlnet.model.sqlnet import SQLNet
import numpy as np
import datetime
import copy

import argparse
import os
from time import time

if __name__ == '__main__':
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
    parser.add_argument('--batchsize', default=64, type=int)
    args = parser.parse_args()
    experiment.log_parameters(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    N_word=300
    B_word=42
    if args.toy:
        USE_SMALL=True
        GPU=True
        BATCH_SIZE=15
    else:
        USE_SMALL=False
        GPU=True
        BATCH_SIZE=args.batchsize
    TRAIN_ENTRY=(True, True, True)  # (AGG, SEL, COND)
    TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
    learning_rate = 1e-4 if args.rl else 1e-3

    # load wikisql
    sql_data, table_data, val_sql_data, val_table_data, test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB = load_dataset(args.dataset, use_small=USE_SMALL)
    
    # concartenate mastercard dummy dataset with wikisql
    dummy_sql_data, dummy_table_data = load_dataset_dummy(0)
    [sql_data.extend(dummy_sql_data) for _ in range(100)]
    table_data.update(dummy_table_data)
    
    # # optinoal: train only on dummy
    # sql_data = dummy_sql_data
    # table_data = dummy_table_data
    
    # train on mc data
    dum_sql_data = dummy_sql_data
    dum_table_data = dummy_table_data
    
    # test model on mc validation data
    dummy_sql_data, dummy_table_data = load_dataset_dummy(0, teststr='_test')
    val_sql_data = dummy_sql_data
    val_table_data = dummy_table_data

    # load word embedding
    tic = time()
    print '==> loading word embedding'
    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), load_used=args.train_emb, use_small=USE_SMALL)
    print 'time to load word emb: ' + str(time() - tic)

    # build sqlnet model
    if not args.baseline:
        tic = time()
        print '==> loading sqlnet constructor'
        model = SQLNet(word_emb, N_word=N_word, use_ca=args.ca, gpu=GPU, trainable_emb = args.train_emb)
        print 'time to load sqlnet constructor: ' + str(time() - tic)
        assert not args.rl, "SQLNet can\'t do reinforcement learning."
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0)

    if args.train_emb:
        agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(args)

    ## transfer learning via finetuning weights from precedent task https://machinelearningmastery.com/transfer-learning-for-deep-learning/
    
    # load pretrained
    savedstr = '_pretrain_wikisql'
    agg_m, sel_m, cond_m = best_model_name(args, savedstr=savedstr)
    # print('==> best model names:', agg_m, sel_m, cond_m)
    # print "Loading from %s"%agg_m
    # model.agg_pred.load_state_dict(torch.load(agg_m))
    # print "Loading from %s"%sel_m
    # model.sel_pred.load_state_dict(torch.load(sel_m))
    # print "Loading from %s"%cond_m
    # model.cond_pred.load_state_dict(torch.load(cond_m))
    # if args.rl or args.train_emb:
    #     print('train_emb is on, so loading best_model')
    #     agg_lm, sel_lm, cond_lm = best_model_name(args, for_load=True)
    #     print "Loading from %s"%agg_lm
    #     model.agg_pred.load_state_dict(torch.load(agg_lm))
    #     print "Loading from %s"%sel_lm
    #     model.sel_pred.load_state_dict(torch.load(sel_lm))
    #     print "Loading from %s"%cond_lm
    #     model.cond_pred.load_state_dict(torch.load(cond_lm))

    ## begin training
    for i in range(100):
        print 'Epoch %d'%(i+1)
        
        # precedent task
        loss_epoch = epoch_train(model, optimizer, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY)
        # finetuned task
        # loss_epoch = epoch_train(model, optimizer, BATCH_SIZE, dum_sql_data, dum_table_data, TRAIN_ENTRY)
        print ' Loss = %s'%loss_epoch
        experiment.log_metric('loss', loss_epoch, step=i)
        
        if not i % 5:
            train_acc_qm = epoch_acc(model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY, epoch=i)
            print ' Train acc_qm: %s\n   breakdown result: %s'%train_acc_qm
            experiment.log_metric('train_acc_qm', train_acc_qm[0], step=i)
        print('==> mc train')
        dum_acc = epoch_acc(model, BATCH_SIZE, dum_sql_data, dum_table_data, TRAIN_ENTRY, write='dummy', experiment=experiment, epoch=i)
        print ' Dum acc_qm: %s\n   breakdown result: %s'%dum_acc
        experiment.log_metric('dummy_acc_qm', dum_acc[0], step=i)

        print('==> mc valid')
        val_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, write='valid', experiment=experiment, epoch=i)
        print ' Dev acc_qm: %s\n   breakdown result: %s'%val_acc
        experiment.log_metric('valid_acc_qm', val_acc[0], step=i)
        
        # agg_m, sel_m, cond_m = best_model_name(args, savedstr='')
        torch.save(model.agg_pred.state_dict(), agg_m)
        torch.save(model.sel_pred.state_dict(), sel_m)
        torch.save(model.cond_pred.state_dict(), cond_m)
