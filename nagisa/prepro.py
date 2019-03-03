# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import

import codecs

import numpy as np

import utils

OOV = utils.OOV
PAD = utils.PAD


def embedding_loader(fn_embedding, word2id, init_range=0.1):
    # load a pre-trained embedding file (word2vec format)
    word2vec = {}
    with codecs.open(fn_embedding, 'r', encoding='utf_8_sig') as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            if i == 0:
                dim_word = int(line.split(' ')[1])
            else:
                line = line.split(' ')
                word = line[0].lower()
                if word in word2id:
                    vec = np.asarray(list(map(float, line[1:])))
                    word2vec[word] = vec

    embs = []
    for word, idx in sorted(word2id.items(), key=lambda x:x[-1]):
        if word.lower() in word2vec:
            embs.append(word2vec[word.lower()])
        else:
            unk = np.random.uniform(-init_range, init_range, dim_word)
            embs.append(unk)
    embs = np.asarray(embs)
    return embs, dim_word


def update_dict(key, dictionary):
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1
    return dictionary


def cut_by_threshold(dictionary, oov, pad, threshold=2):
    token2id = {oov:0, pad:1}
    for token, freq in reversed(sorted(dictionary.items(), key=lambda x:x[1])):
        if freq >= threshold:
            token2id[token] = len(token2id)
    return token2id


def create_vocabs_from_trainset(trainset, threshold=2,
                                fn_dictionary=None, save_vocabs=True,
                                fn_vocabs=None, delimiter='\t',
                                newline='EOS', oov=OOV, pad=PAD):
    # Creat a word-to-POStags dictionary.
    word2postags = {}
    if fn_dictionary is not None:
        with codecs.open(fn_dictionary, 'r', encoding='utf_8_sig') as texts:
            for text in texts:
                text = utils.utf8rstrip(text)
                word, postag = text.split('\t')
                word = utils.normalize(word)
                # lower setting: 1
                word = word.lower()
                if word in word2postags:
                    word2postags[word].append(postag)
                else:
                    word2postags[word] = [postag]

    # Creat a word-to-index dictionary and a index-to-word dictionary.
    dictionary = {oov:0, pad:1}
    for word in word2postags.keys():
        dictionary[word] = len(dictionary)
    id2word = {i:w for w, i in dictionary.items()}

    # Creat a unigram-to-index dictionary, a bigram-to-index dictionary.
    # Reconstruct a word-to-index dictionary.
    words   = []
    uni2id  = {}
    bi2id   = {}
    word2id = {}
    pos2id  = {oov:0}
    with codecs.open(trainset, 'r', encoding='utf_8_sig') as texts:
        for text in texts:
            text = utils.utf8rstrip(text)
            if text == newline:
                sent = ''.join(words)
                unis = utils.get_unigram(sent)
                for uni in unis:
                    uni2id = update_dict(uni, uni2id)

                bis = utils.get_bigram(sent)
                for bi in bis:
                    bi2id = update_dict(bi, bi2id)

                words_at_i = utils.get_words_starting_at_i(sent, dictionary)
                words_at_i += utils.get_words_ending_at_i(sent, dictionary)
                for words in words_at_i:
                    for wid in words:
                        word    = id2word[wid]
                        word2id = update_dict(word, word2id)
                words = []

            else:
                word, pos = text.split(delimiter)
                word = utils.normalize(word)
                word = word.replace(' ', '　')
                # lower setting: 2
                word = word.lower()
                words.append(word)
                word2id = update_dict(word, word2id)
                pos2id  = update_dict(pos, pos2id)

    # Cut keys by frequency threshold.
    uni2id  = cut_by_threshold(uni2id,  oov, pad, threshold)
    bi2id   = cut_by_threshold(bi2id,   oov, pad, threshold)
    word2id = cut_by_threshold(word2id, oov, pad, threshold)

    # Creat a POStag-to-index dictionary.
    pos2id = {k:i for i,k in enumerate(pos2id.keys())}
    word2postags = {k:[pos2id[p] if p in pos2id else pos2id[oov] for p in list(set(v))]\
                    for k,v in word2postags.items()}

    vocabs = [uni2id, bi2id, word2id, pos2id, word2postags]
    if save_vocabs is True:
        utils.dump_data(vocabs, fn_vocabs)

    return vocabs


class from_file(object):
    def __init__(self, filename, window_size, vocabs,
                 newline='EOS', delimiter='\t'):
        self.words    = []
        self.ws_data  = []
        self.pos_data = []
        self.filename = filename
        uni2id, bi2id, word2id, pos2id, word2postags = vocabs

        if u'名詞' in pos2id:
            self.use_noun_heuristic = True
        else:
            self.use_noun_heuristic = False

        with codecs.open(filename, 'r', encoding='utf_8_sig') as texts:
            wids  = [] # Word index
            cids  = [] # Character index
            pids  = [] # POStag index
            words = [] # Original Words
            ptags = [] # Original POStags
            for text in texts:
                text = utils.utf8rstrip(text)
                if text == newline:
                    sent = ''.join(words)
                    segmented_sent = ' '.join(words)
                    tags = utils.make_tags_as_bmes(segmented_sent)

                    feats = utils.feature_extraction(text=sent,
                                                     uni2id=uni2id,
                                                     bi2id=bi2id,
                                                     dictionary=word2id,
                                                     window_size=window_size)

                    self.words.append(words)
                    self.ws_data.append([feats, tags])
                    self.pos_data.append([[cids, wids, ptags], pids])
                    # reset index lists
                    wids  = []
                    cids  = []
                    pids  = []
                    words = []
                    ptags = []

                else:
                    word, pos = text.split(delimiter)
                    word = utils.normalize(word)
                    word = word.replace(' ', '　')
                    # lower setting: 3
                    word = word.lower()

                    if word in word2postags:
                        w2p = word2postags[word]
                    else:
                        w2p = [0] # OOV
                    if self.use_noun_heuristic is True:
                        if word.isalnum() is True:
                            if w2p == [0]:
                                w2p = [pos2id[u'名詞']]
                            else:
                                w2p.append(pos2id[u'名詞'])

                    w2p = list(set(w2p))
                    ptags.append(w2p)

                    words.append(word)
                    if word in word2id:
                        wids.append(word2id[word])
                    else:
                        wids.append(word2id[OOV])

                    if pos in pos2id:
                        pids.append(pos2id[pos])
                    else:
                        pids.append(pos2id[OOV])

                    char_seq = []
                    for char in word:
                        if char in uni2id:
                            char_seq.append(uni2id[char])
                        else:
                            char_seq.append(uni2id[OOV])
                    cids.append(char_seq)
