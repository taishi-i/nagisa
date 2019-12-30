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

    def __init__(self, vocabs=None, params=None, hp=None, single_word_list=None):
        if vocabs is None:
            vocabs = base + '/data/nagisa_v001.dict'
        if params is None:
            params = base + '/data/nagisa_v001.model'
        if hp is None:
            hp = base + '/data/nagisa_v001.hp'

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
            single_word_list = [w.replace('(', '\(').replace(')', '\)')
                                for w in single_word_list]
            single_word_list = sorted(single_word_list, key=lambda x:-len(x))
            if len(single_word_list) > 0:
                self.pattern = re.compile('|'.join(single_word_list))

        # If use_noun_heuristic is True, nouns are more lilely to appear.
        if u'名詞' in self._pos2id:
            self.use_noun_heuristic = True
        else:
            self.use_noun_heuristic = False


    def wakati(self, text, lower=False):
        """Word segmentation function. Return the segmented words.

        args:
            - text (str): An input sentence.
            - lower (bool): If lower is True, all uppercase characters in a list \
                            of the words are converted into lowercase characters.

        return:
            - words (list): A list of the words.
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

                if (span_e - span_s) == 1:
                    tags[span_s:span_e] = [3]
                else:
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


    def _postagging(self, words, lower=False):
        if lower is True:
            words = [w.lower() for w in words]

        wids = utils.conv_tokens_to_ids(words, self._word2id)
        cids = [utils.conv_tokens_to_ids([c for c in w], self._uni2id) for w in words]
        tids = []
        for w in words:
            if w in self._word2postags:
                w2p = self._word2postags[w]
            else:
                w2p = [0]
            if self.use_noun_heuristic is True:
                if w.isalnum() is True:
                    if w2p == [0]:
                        w2p = [self._pos2id[u'名詞']]
                    else:
                        w2p.append(self._pos2id[u'名詞'])
            w2p = list(set(w2p))
            tids.append(w2p)

        X = [cids, wids, tids]
        postags = [self._id2pos[pid] for pid in self._model.POStagging(X)]
        return postags


    def postagging(self, words, lower=False):
        """ Return the words with POS-tags of the given words.

        args:
            - words (list): Input words.
            - lower (bool): If lower is True, all uppercase characters in a list \
                            of the words are converted into lowercase characters.
        return:
            - object : The object of the words with POS-tags.
        """
        return self.decode(words, lower)


    def decode(self, words, lower=False):
        """ Return the words with tags of the given words.

        args:
            - words (list): Input words.
            - lower (bool, optional): If lower is True, all uppercase characters in a list \
                            of the words are converted into lowercase characters.
        return:
            - object : The object of the words with tags.
        """
        if not type(words) == list:
            raise AssertionError("Please input a list of words.")
        words = [utils.preprocess_without_rstrip(w) if w == " " or w == "　"
                 else utils.preprocess(w) for w in words]
        postags = self._postagging(words, lower)
        return postags


    def tagging(self, text, lower=False):
        """ Return the words with POS-tags of the given sentence.

        args:
            - text (str): An input sentence.
            - lower (bool): If lower is True, all uppercase characters in a list \
                            of the words are converted into lowercase characters.
        return:
            - object : The object of the words with POS-tags.
        """
        return self._Token(text, lower, self.wakati, self._postagging)


    def filter(self, text, lower=False, filter_postags=None):
        """Return the filtered words with POS-tags of the given sentence.

        args:
            - text (str): An input sentence.
            - lower (bool): If lower is True, all uppercase characters in a list \
                            of the words are converted into lowercase characters.
            - filter_postags (list): Filtering the word with the POS-tag in \
                                     filter_postags from a text.

        return:
            - object : The object of the words with POS-tags.
        """
        if filter_postags is None:
            filter_postags = []

        words   = []
        postags = []
        tokens  = self.tagging(text, lower)
        for word, postag in zip(tokens.words, tokens.postags):
            if not postag in filter_postags:
                words.append(word)
                postags.append(postag)
        return self._Token(text, lower, self.wakati, self._postagging,
                           _words=words, _postags=postags)


    def extract(self, text, lower=False, extract_postags=None):
        """Return the extracted words with POS-tags of the given sentence.

        args:
            - text (str): An input sentence.
            - lower (bool): If lower is True, all uppercase characters in a list \
                            of the words are converted into lowercase characters.
            - filter_postags (list): Extracting the word with the POS-tag in \
                                     filter_postags from a text.

        return:
            - object : The object of the words with POS-tags.

        """
        if extract_postags is None:
            extract_postags = []

        words   = []
        postags = []
        tokens  = self.tagging(text, lower)
        for word, postag in zip(tokens.words, tokens.postags):
            if postag in extract_postags:
                words.append(word)
                postags.append(postag)
        return self._Token(text, lower, self.wakati, self._postagging,
                           _words=words, _postags=postags)


    class _Token(object):
        def __init__(self, text, lower, wakati, postagging, _words=None, _postags=None):
            self.text = text
            self.__lower = lower
            self.__words = _words
            self.__postags = _postags
            self.__wakati = wakati
            self.__postagging = postagging

        @property
        def words(self):
            if self.__words is None:
                self.__words = self.__wakati(self.text, self.__lower)
            return self.__words

        @property
        def postags(self):
            if self.__postags is None:
                self.__postags = self.__postagging(self.words, self.__lower)
            return self.__postags

        def __str__(self):
            return ' '.join([w+'/'+p for w, p in zip(self.words, self.postags)])
