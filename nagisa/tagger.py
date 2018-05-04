# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import re
import sys
import utils
import nagisa.model as model

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)


class Tagger(object):
    """
    This class has a word segmentation function and a POS-tagging function for Japanese.
    """
    def __init__(self, vocabs=base+'/data/nagisa_v001.dict',
                 params=base+'/data/nagisa_v001.model', 
                 hp=base+'/data/nagisa_v001.hp',
                 single_word_list=None):
        # Load vocaburary files
        vocabs = utils.load_data(vocabs) 
        self._uni2id, self._bi2id, self._word2id, self._pos2id, self._word2postags = vocabs
        self._id2pos = {v:k for k, v in self._pos2id.items()}
        self.id2pos  = self._id2pos
        self.postags = [postag for postag in self._pos2id.keys()]
        # Load a hyper-parameter file
        self._hp = utils.load_data(hp)
        # Construct a word segmentation model and a pos tagging model
        self._model = model.Model(self._hp, params)

        # If a word is included in the single_word_list,
        # it is recognized as a single word forcibly.
        self.pattern = None
        if single_word_list:
            single_word_list = [utils.preprocess(w) for w in single_word_list if len(w) > 1] 
            if len(single_word_list) > 0:
                self.pattern = re.compile('|'.join(single_word_list))


    def wakati(self, text, lower=False):
        """
        Return the words of the given sentence.
        Input: str (a sentence)
        Output: the list of the words
        """
        text = utils.preprocess(text)
        lower_text = text.lower()
        feats = utils.feature_extraction(text=lower_text,
                                         uni2id=self._uni2id,
                                         bi2id=self._bi2id,
                                         dictionary=self._word2id,
                                         window_size=self._hp['WINDOW_SIZE'])
        obs  = self._model.encode_ws(feats)
        obs  = [ob.npvalue() for ob in obs] 
        tags = utils.np_viterbi(self._model.trans_array, obs)

        # A word can be recognized as a single word forcibly.
        if self.pattern:
            for match in self.pattern.finditer(text):
                span = match.span()
                span_s = span[0]
                span_e = span[1]
                tags[span_s:span_e] = [0]+[1]*((span_e-span_s)-2)+[2]

                if span_s != 0:
                    previous_tag = tags[span_s-1]
                    if previous_tag == 0:   # 0 is BEGIN tag
                        tags[span_s-1] = 3  # 3 is SINGLE tag
                    elif previous_tag == 1: # 1 is MIDDEL tag
                        tags[span_s-1] = 2  # 2 is END tag

                if span_e != len(text):
                    next_tag = tags[span_e]
                    if next_tag == 1:    # 1 is MIDDEL tag
                        tags[span_e] = 0 # 0 is BEGIN tag
                    elif next_tag == 2:  # 2 is END tag
                        tags[span_e] = 3 # 3 is SINGLE tag

        if lower is True:
            words = utils.segmenter_for_bmes(lower_text, tags)
        else:
            words = utils.segmenter_for_bmes(text, tags)
        return words


    def tagging(self, text, lower=False):
        """
        Return the words with POS-tags of the given sentence.
        Input: str (a sentence)
        Output: the object of the words with POS-tags
        """
        words = self.wakati(text, lower)

        wids = utils.conv_tokens_to_ids(words, self._word2id)
        cids = [utils.conv_tokens_to_ids([c for c in w.lower()], self._uni2id) for w in words]
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


    def filter(self, text, lower=False, filter_postags=[]):
        """
        Return the filtered words with POS-tags of the given sentence.
        Input: str (a sentence)
        Output: the object of the words with POS-tags
        """
        words   = []
        postags = []
        tokens  = self.tagging(text, lower)
        for word, postag in zip(tokens.words, tokens.postags):
            if not postag in filter_postags:
                words.append(word)
                postags.append(postag)
        return self._Token(text, words, postags)


    def extract(self, text, lower=False, extract_postags=[]):
        """
        Return the extracted words with POS-tags of the given sentence.
        Input: str (a sentence)
        Output: the object of the words with POS-tags
        """
        words   = []
        postags = []
        tokens  = self.tagging(text, lower)
        for word, postag in zip(tokens.words, tokens.postags):
            if postag in extract_postags:
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
