# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import

import time
import utils
import model
import random
import prepro
import argparse
import subprocess
import numpy as np
import dynet as dy

import mecab_system_eval
from tagger import Tagger

parser = argparse.ArgumentParser()
parser.add_argument('-train',  type=str)
parser.add_argument('-dev',    type=str)
parser.add_argument('-test',   type=str)
parser.add_argument('-dict',   type=str)
parser.add_argument('-output', type=str, default='nagisa_v001')
args = parser.parse_args()


def main():
    """
    usage: python train.py -train <train_file> -dev <dev_file> -test <test_file> \
                           -dict <dictionary_file> -output <model_name>

    file format (train_file, dev_file, test_file):

        word\tPOS-tag
        word\tPOS-tag
        word\tPOS-tag
        EOS
        word\tPOS-tag
        word\tPOS-tag
        ...
        word\tPOS-tag
        EOS
        

    file format (dictionary_file):
        
        word\tPOS-tag
        word\tPOS-tag
        word\tPOS-tag
        ...
    
    """

    # Set hyperparameters
    hp = {'LAYERS':1,
          'THRESHOLD':2, 
          'DECAY':3, 
          'EPOCH':25, 
          'WINDOW_SIZE':3, 
          'DIM_UNI':32, 
          'DIM_BI':16,
          'DIM_WORD':16,
          'DIM_CTYPE':8,
          'DIM_TAGEMB':16,
          'DIM_HIDDEN':100,
          'LEARNING_RATE':0.075,
          'DROPOUT_RATE':0.2,

          'TRAINSET':args.train,
          'TESTSET':args.test,
          'DEVSET':args.dev,
          'DICTIONARY':args.dict,

          'HYPERPARAMS':'data/'+args.output+'.hp',
          'MODEL':'data/'+args.output+'.model',
          'VOCAB':'data/'+args.output+'.dict',

          'EPOCH_MODEL':'data/epoch.model'}

    # Setup vocabuary files
    vocabs = prepro.create_vocabs_from_trainset(trainset=hp['TRAINSET'],
                                                fn_dictionary=hp['DICTIONARY'],
                                                fn_vocabs=hp['VOCAB'])

    # Update hyper-parameters
    hp['VOCAB_SIZE_UNI']    = len(vocabs[0])
    hp['VOCAB_SIZE_BI']     = len(vocabs[1])
    hp['VOCAB_SIZE_WORD']   = len(vocabs[2])
    hp['VOCAB_SIZE_POSTAG'] = len(vocabs[3])
    # Preprocess
    ws = hp['WINDOW_SIZE']
    TrainData = prepro.from_file(filename=hp['TRAINSET'], window_size=ws, vocabs=vocabs)
    TestData  = prepro.from_file(filename=hp['TESTSET'],  window_size=ws, vocabs=vocabs)
    DevData   = prepro.from_file(filename=hp['DEVSET'],   window_size=ws, vocabs=vocabs)
    # Construct networks
    _model = model.Model(hp=hp)
    # Start training
    fit(hp, model=_model, train_data=TrainData, test_data=TestData, dev_data=DevData)


def evaluation(hp, fn_model, data):
    tagger = Tagger(vocabs=hp['VOCAB'], params=fn_model, hp=hp['HYPERPARAMS'])

    def data_for_eval(words, postags):
        sent = []
        for w, p in zip(words, postags):
            p = w+"\t"+p
            if mecab_system_eval.PY_3 is True:
                w = w.encode("UTF-8")
                p = p.encode("UTF-8")
            sent.append([w, p])
        return sent

    sys_data = []
    ans_data = []
    indice   = [i for i in range(len(data.ws_data))]
    for i in indice:
        words   = data.words[i]
        pids    = data.pos_data[i][1]
        postags = [tagger.id2pos[pid] for pid in pids]
        ans_data.append(data_for_eval(words, postags))

        output      = tagger.tagging(''.join(words))
        sys_words   = output.words
        sys_postags = output.postags
        sys_data.append(data_for_eval(sys_words, sys_postags))

    r = mecab_system_eval.mecab_eval(sys_data, ans_data)
    _, _, ws_f, _, _, pos_f = mecab_system_eval.calculate_fvalues(r)
    return ws_f, pos_f


def fit(hp, model, train_data, test_data, dev_data):
    for k, v in hp.items():
        print("[nagisa] "+k+":\t", v)

    logs = ['Epoch', 'LR', 'Loss', 'Time(m)', 'DevWS', 'DevPOS', 'TestWS', 'TestPOS']
    print('\n'+'\t'.join(logs))
    utils.dump_data(hp, hp['HYPERPARAMS'])

    decay_counter  = 0
    best_dev_score = 0.
    indice = [i for i in range(len(train_data.ws_data))]
    for e in range(hp['EPOCH']):
        t = time.time()
        losses = 0.
        random.shuffle(indice)
        for i in indice:
            # Word Segmentation
            X = train_data.ws_data[i][0]
            Y = train_data.ws_data[i][1]
            obs = model.encode_ws(X, train=True)
            gold_score = model.score_sentence(obs, Y)
            forward_score = model.forward(obs)
            loss = forward_score-gold_score
            # Update
            loss.backward()
            model.trainer.update()
            losses += loss.value()

            # POS-tagging
            X = train_data.pos_data[i][0]
            Y = train_data.pos_data[i][1]
            loss = model.get_POStagging_loss(X, Y)
            losses += loss.value()
            # Update
            loss.backward()
            model.trainer.update()

        model.model.save(hp['EPOCH_MODEL'])
        dev_ws_f, dev_pos_f = evaluation(hp, fn_model=hp['EPOCH_MODEL'], data=dev_data)

        if dev_ws_f > best_dev_score:
            best_dev_score = dev_ws_f
            decay_counter = 0
            model.model.save(hp['MODEL'])
            test_ws_f, test_pos_f = evaluation(hp, fn_model=hp['MODEL'], data=test_data)
        else:
            decay_counter += 1
            if decay_counter >= hp['DECAY']:
                model.trainer.learning_rate = model.trainer.learning_rate/2
                decay_counter = 0

        logs = [e, model.trainer.learning_rate, losses, (time.time()-t)/60,
                dev_ws_f, dev_pos_f, test_ws_f, test_pos_f]
        print('\t'.join([log[:5] for log in map(str, logs)]))


if __name__ == '__main__':
    main()
