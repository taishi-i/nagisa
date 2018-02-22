# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import sys
import utils
import nagisa.model as model

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)

class Tagger(object):
    def __init__(self, vocabs=base+'/data/nagisa_v001.dict',
                 params=base+'/data/nagisa_v001.model', 
                 hp=base+'/data/nagisa_v001.hp'):
        # Load vocaburary files
        vocabs = utils.load_data(vocabs) 
        self._uni2id, self._bi2id, self._word2id, self._pos2id, self._word2postags = vocabs
        self._id2pos = {v:k for k, v in self._pos2id.items()}
        self.id2pos = self._id2pos
        self.postags = [postag for postag in self._pos2id.keys()]
        # Load a hyper-parameter file
        self._hp = utils.load_data(hp)
        # Construct a word segmentation model and a pos tagging model
        self._model = model.Model(self._hp, params)


    def wakati(self, text):
        """
        Return the words of the given sentence.
        Input: str (a sentence)
        Output: the list of the words
        """
        text = utils.utf8rstrip(text)
        text = utils.normalize(text)                                            
        text = text.replace(' ', '　')
        feats = utils.feature_extraction(text=text.lower(), 
                                         uni2id=self._uni2id,
                                         bi2id=self._bi2id,
                                         dictionary=self._word2id,
                                         window_size=self._hp['WINDOW_SIZE'])
        obs   = self._model.encode_ws(feats)
        obs   = [ob.npvalue() for ob in obs] 
        tags  = utils.np_viterbi(self._model.trans_array, obs)
        words = utils.segmenter_for_bmes(text, tags)
        return words


    def tagging(self, text):
        """
        Return the words with POS tags of the given sentence.
        Input: str (a sentence)
        Output: the object of the words with POS tags
        """
        words = self.wakati(text)

        wids = utils.conv_tokens_to_ids(words, self._word2id)
        cids = [utils.conv_tokens_to_ids([c for c in w], self._uni2id) for w in words]
        tids = []
        for w in words:
            w = w.lower()
            if w in self._word2postags:
                w2p = self._word2postags[w]
            else:
                w2p = [0]
            if w.isalnum() is True:
                if w2p == [0]:
                    w2p = [self._pos2id[u'名詞']]
                else:
                    w2p.append(self._pos2id[u'名詞'])
            w2p = list(set(w2p))
            tids.append(w2p)

        X = [cids, wids, tids]
        postags = [self._id2pos[pid] for pid in self._model.POStagging(X)]
        return self._Token(text, words, postags)


    def filter(self, text, filter_postags=[]):
        words = []
        postags = []
        tokens = self.tagging(text)
        for word, postag in zip(tokens.words, tokens.postags):
            if not postag in filter_postags:
                words.append(word)
                postags.append(postag)
        return self._Token(text, words, postags)


    def extract(self, text, filter_postags=[]):
        words = []
        postags = []
        tokens = self.tagging(text)
        for word, postag in zip(tokens.words, tokens.postags):
            if postag in filter_postags:
                words.append(word)
                postags.append(postag)
        return self._Token(text, words, postags)


    class _Token(object):
        def __init__(self, text, words, postags):
            self.text = text
            self.words = words
            self.postags = postags

        def __str__(self):
            return ' '.join([w+'/'+p for w, p in zip(self.words, self.postags)])
