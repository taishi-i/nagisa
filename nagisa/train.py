# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import

import time
import random
import logging
from collections import OrderedDict

import utils
import model
import prepro
import mecab_system_eval
from tagger import Tagger

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def fit(train_file, dev_file, test_file, model_name,
        dict_file=None, emb_file=None, delimiter='\t', newline='EOS',
        layers=1, min_count=3, decay=1, epoch=10, window_size=3,
        dim_uni=32, dim_bi=16, dim_word=16, dim_ctype=8, dim_tagemb=16,
        dim_hidden=100, learning_rate=0.1, dropout_rate=0.3, seed=1234):

    """Train a joint word segmentation and sequence labeling (e.g, POS-tagging, NER) model.

    args:
        - train_file (str): Path to a train file.
        - dev_file (str): Path to a development file for early stopping.
        - test_file (str):  Path to a test file for evaluation.
        - model_name (str): Output model filename.
        - dict_file (str, optional): Path to a dictionary file.
        - emb_file (str, optional): Path to a pre-trained embedding file (word2vec format).
        - delimiter (str, optional): Separate word and tag in each line by 'delimiter'.
        - newline (str, optional):  Separate lines in the file by 'newline'.
        - layers (int, optional): RNN Layer size.
        - min_count (int, optional): Ignores all words with total frequency lower than this.
        - decay (int, optional): Learning rate decay.
        - epoch (int, optional): Epoch size.
        - window_size (int, optional): Window size of the context characters for word segmentation.
        - dim_uni (int, optional): Dimensionality of the char-unigram vectors.
        - dim_bi (int, optional): Dimensionality of the char-bigram vectors.
        - dim_word (int, optional): Dimensionality of the word vectors.
        - dim_ctype (int, optional): Dimensionality of the character-type vectors.
        - dim_tagemb (int, optional): Dimensionality of the tag vectors.
        - dim_hidden (int, optional): Dimensionality of the BiLSTM's hidden layer.
        - learning_rate (float, optional): Learning rate of SGD.
        - dropout_rate (float, optional): Dropout rate of the input vector for BiLSTMs.
        - seed (int, optional): Random seed.

    return:
        - Nothing. After finish training, however,
          save the three model files (*.vocabs, *.params, *.hp) in the current directory.

    """

    random.seed(seed)

    hp = OrderedDict({
          'LAYERS':layers,
          'THRESHOLD':min_count,
          'DECAY':decay,
          'EPOCH':epoch,
          'WINDOW_SIZE':window_size,
          'DIM_UNI':dim_uni,
          'DIM_BI':dim_bi,
          'DIM_WORD':dim_word,
          'DIM_CTYPE':dim_ctype,
          'DIM_TAGEMB':dim_tagemb,
          'DIM_HIDDEN':dim_hidden,
          'LEARNING_RATE':learning_rate,
          'DROPOUT_RATE':dropout_rate,
          'SEED': seed,

          'TRAINSET':train_file,
          'TESTSET':test_file,
          'DEVSET':dev_file,
          'DICTIONARY':dict_file,
          'EMBEDDING':emb_file,

          'HYPERPARAMS':model_name+'.hp',
          'MODEL':model_name+'.params',
          'VOCAB':model_name+'.vocabs',
          'EPOCH_MODEL':model_name+'_epoch.params'
    })


    # Preprocess
    vocabs = prepro.create_vocabs_from_trainset(trainset=hp['TRAINSET'],
                                                fn_dictionary=hp['DICTIONARY'],
                                                fn_vocabs=hp['VOCAB'],
                                                delimiter=delimiter,
                                                newline=newline)

    if emb_file is not None:
        embs, dim_word = prepro.embedding_loader(fn_embedding=hp['EMBEDDING'],
                                                 word2id=vocabs[2])
        hp['DIM_WORD'] = dim_word
    else:
        embs = None


    TrainData = prepro.from_file(filename=hp['TRAINSET'],
                                 window_size=hp['WINDOW_SIZE'],
                                 vocabs=vocabs,
                                 delimiter=delimiter,
                                 newline=newline)
    TestData  = prepro.from_file(filename=hp['TESTSET'],
                                 window_size=hp['WINDOW_SIZE'],
                                 vocabs=vocabs,
                                 delimiter=delimiter,
                                 newline=newline)
    DevData   = prepro.from_file(filename=hp['DEVSET'],
                                 window_size=hp['WINDOW_SIZE'],
                                 vocabs=vocabs,
                                 delimiter=delimiter,
                                 newline=newline)

    # Update hyper-parameters
    hp['NUM_TRAIN']         = len(TrainData.ws_data)
    hp['NUM_TEST']          = len(TestData.ws_data)
    hp['NUM_DEV']           = len(DevData.ws_data)
    hp['VOCAB_SIZE_UNI']    = len(vocabs[0])
    hp['VOCAB_SIZE_BI']     = len(vocabs[1])
    hp['VOCAB_SIZE_WORD']   = len(vocabs[2])
    hp['VOCAB_SIZE_POSTAG'] = len(vocabs[3])

    # Construct networks
    _model = model.Model(hp=hp, embs=embs)

    # Start training
    _start(hp, model=_model, train_data=TrainData, test_data=TestData, dev_data=DevData)


def _evaluation(hp, fn_model, data):
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


def _start(hp, model, train_data, test_data, dev_data):
    for k, v in hp.items():
        logging.info('[nagisa] {}: {}'.format(k, v))

    logs = '{:5}\t{:5}\t{:5}\t{:5}\t{:8}\t{:8}\t{:8}\t{:8}'.format(
        'Epoch', 'LR', 'Loss', 'Time_m', 'DevWS_f1',
        'DevPOS_f1', 'TestWS_f1', 'TestPOS_f1')
    logging.info(logs)


    utils.dump_data(hp, hp['HYPERPARAMS'])

    decay_counter  = 0
    best_dev_score = -1.0
    indice = [i for i in range(len(train_data.ws_data))]
    for e in range(1, hp['EPOCH']+1):
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
        dev_ws_f, dev_pos_f = _evaluation(hp, fn_model=hp['EPOCH_MODEL'], data=dev_data)

        if dev_ws_f > best_dev_score:
            best_dev_score = dev_ws_f
            decay_counter = 0
            model.model.save(hp['MODEL'])
            test_ws_f, test_pos_f = _evaluation(hp, fn_model=hp['MODEL'], data=test_data)
        else:
            decay_counter += 1
            if decay_counter >= hp['DECAY']:
                model.trainer.learning_rate = model.trainer.learning_rate/2
                decay_counter = 0

        losses = losses/len(indice)
        logs = [e, model.trainer.learning_rate, losses, (time.time()-t)/60,
                dev_ws_f, dev_pos_f, test_ws_f, test_pos_f]

        logs = [log[:5] for log in map(str, logs)]

        logs = '{:5}\t{:5}\t{:5}\t{:5}\t{:8}\t{:8}\t{:8}\t{:8}'.format(
            logs[0], logs[1], logs[2], logs[3], logs[4],
            logs[5], logs[6], logs[7])

        logging.info(logs)

