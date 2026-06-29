# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import

import re
import sys
import gzip
import codecs
import numpy as np
import unicodedata

cimport cython
from libc.math cimport expf, tanhf

from six.moves import cPickle

reload(sys)
if sys.version_info.major == 2:
    sys.setdefaultencoding('utf-8')

cdef unicode __OOV = u'oov'
cdef unicode __PAD = u'pad'

OOV = __OOV
PAD = __PAD

_hiragana = re.compile(u'[\u3040-\u309F]')
_katakana = re.compile(u'[\u30A1-\u30FA]')
_kanji    = re.compile(u'[\u4e00-\u9fa5]')
_alpha    = re.compile(u'[a-zA-Z]')
_numeric  = re.compile(u'[0-9]')


cpdef unicode utf8rstrip(text):
    if type(text) != unicode:
        return unicode(text.rstrip(), 'utf-8')
    else:
        return text.rstrip()


cpdef unicode normalize(unicode text):
    return unicodedata.normalize('NFKC', text)


cpdef unicode preprocess(text):
    text = utf8rstrip(text)
    text = normalize(text)
    text = text.replace('İ', 'I')
    text = text.replace(' ', '　')
    return text


cpdef unicode preprocess_without_rstrip(text):
    if type(text) != unicode:
        text = unicode(text, 'utf-8')
    text = normalize(text)
    text = text.replace('İ', 'I')
    text = text.replace(' ', '　')
    return text


cpdef list get_unigram(unicode text):
    cdef unicode uni
    return [uni for uni in text]


cpdef list get_bigram(unicode text):
    cdef:
        int i
        int length_text = len(text)
        unicode end_symbol = u'<E>'
    return [text[i]+end_symbol if i == length_text-1 else text[i:i+2]
            for i in range(length_text)]


cpdef int get_chartype(unicode character):
    if _hiragana.search(character):
        return 0
    elif _katakana.search(character):
        return 1
    elif _kanji.search(character):
        return 2
    elif _alpha.search(unicodedata.normalize('NFKC', character)):
        return 3
    elif _numeric.search(unicodedata.normalize('NFKC', character)):
        return 4
    else:
        return 5


cpdef list get_words_starting_at_i(unicode text, dict dictionary):
    cdef:
        int i
        int j
        int length_text = len(text)
        list subwords
        list words_starting_at_i = []
        unicode sub

    for i in range(length_text):
        subwords = []
        for j in range(i, min(i+8, length_text)):
            sub = text[i:j+1]
            if sub in dictionary:
                subwords.append(dictionary[sub])
        if len(subwords) == 0:
            subwords.append(dictionary[__OOV])
        words_starting_at_i.append(subwords)
    return words_starting_at_i


cpdef list get_words_ending_at_i(unicode text, dict dictionary):
    cdef:
        int i
        int j
        int length_text = len(text)
        list subwords
        list words_ending_at_i = []

    text = text[::-1]
    for i in range(length_text):
        subwords = []
        for j in range(i, min(i+8, length_text)):
            sub = text[i:j+1][::-1]
            if sub in dictionary:
                subwords.append(dictionary[sub])
        if len(subwords) == 0:
            subwords.append(dictionary[__OOV])
        words_ending_at_i.append(subwords)
    return words_ending_at_i[::-1]


cpdef list conv_tokens_to_ids(list words, dict word2id):
    cdef unicode word
    return [word2id[word] if word in word2id else word2id[__OOV] for word in words]


cpdef list context_window(list l, int win, int pad_id=1):
    cdef:
        int length_l = len(l)

    assert (win % 2) == 1
    assert win >=1
    lpadded = int(win/2) * [pad_id] + l + int(win/2) * [pad_id]
    out = [lpadded[i:i+win] for i in range(length_l)]
    assert len(out) == len(l)
    return out


cpdef list feature_extraction(unicode text, dict uni2id, dict bi2id,
                              dict dictionary, int window_size):
    # character-level features
    unigrams = get_unigram(text)
    bigrams = get_bigram(text)
    uids = context_window(conv_tokens_to_ids(unigrams, uni2id), window_size)
    bids = context_window(conv_tokens_to_ids(bigrams, bi2id), window_size)
    cids = context_window([get_chartype(uni) for uni in unigrams], window_size, pad_id=6)

    # word-level features
    wids_s = get_words_starting_at_i(text, dictionary)
    wids_e = get_words_ending_at_i(text, dictionary)

    features = [uids, bids, cids, wids_s, wids_e]
    return features


cpdef dict load_dictionary(dict_path):
    cdef dict word_dict = {__OOV:0, __PAD:1}
    with open(dict_path, 'r') as words:
        for word in words:
            word = utf8rstrip(word)
            if not word in word_dict:
                word_dict[word] = len(word_dict)
    return word_dict


cpdef list make_tags_as_bmes(unicode text):
    cdef:
        int i
        int len_word
        list tags = []
        list words = text.split(u' ')
    for word in words:
        len_word = len(word)
        if len_word < 2:
            tags += [3]
        elif len_word == 2:
            tags += [0, 2]
        elif len_word > 2:
            for i in range(len_word):
                if i == 0:
                    tags += [0]
                elif i == len_word-1:
                    tags += [2]
                else:
                    tags += [1]
    assert len(''.join(words)) == len(tags)
    return tags


cpdef list segmenter_for_bmes(unicode chars, list tags):
    cdef:
        int tag
        list words = []
        unicode partical_word = u''
        unicode character

    assert len(chars) == len(tags)
    for character, tag in zip(chars, tags):
        if tag == 3:
            if partical_word is not u'':
                words.append(partical_word)
                partical_word = u''
            words.append(character)
        elif tag == 2:
            partical_word += character
            words.append(partical_word)
            partical_word = u''
        else:
            partical_word += character
    if partical_word is not u'':
        words.append(partical_word)
    return words


cpdef dump_data(data, fn):
    with gzip.open(fn, 'wb') as gf:
        cPickle.dump(data, gf, protocol=2)
        gf.close()


cpdef load_data(fn):
    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _lstm_direction(float[:, ::1] XG, float[:, ::1] WhT,
                          float[::1] g, float[::1] h, float[::1] c,
                          float[:, ::1] H, int col_off, bint reverse):
    """One LSTM direction over precomputed inputs XG[t] = Wx*x_t + b
    (already includes the forget-gate bias). Gate columns are packed as
    [i, f, o, g], DyNet VanillaLSTM semantics. Writes hidden states into
    H[:, col_off:col_off+h]; for the backward direction (reverse=True)
    XG is over the reversed input and outputs land at T-1-t."""
    cdef:
        int T = XG.shape[0]
        int h4 = XG.shape[1]
        int hd = h4 // 4
        int t, row, j, k
        float hk, ct

    for j in range(hd):
        h[j] = 0
        c[j] = 0

    for t in range(T):
        for j in range(h4):
            g[j] = XG[t, j]
        for k in range(hd):
            hk = h[k]
            for j in range(h4):
                g[j] += hk * WhT[k, j]
        row = T - 1 - t if reverse else t
        for j in range(hd):
            ct = ((1.0 / (1.0 + expf(-g[hd + j]))) * c[j]
                  + (1.0 / (1.0 + expf(-g[j]))) * tanhf(g[3 * hd + j]))
            c[j] = ct
            h[j] = (1.0 / (1.0 + expf(-g[2 * hd + j]))) * tanhf(ct)
            H[row, col_off + j] = h[j]


def birnn_transduce(XGf, XGb, WhTf, WhTb):
    """Fused BiLSTM layer recurrence: returns H (T, 2h) float32 where
    H[t] = [forward state at t, backward state for position t]."""
    cdef int T = XGf.shape[0]
    cdef int hd = WhTf.shape[0]
    H_arr = np.empty((T, 2 * hd), dtype=np.float32)
    cdef float[:, ::1] H = H_arr
    cdef float[::1] g = np.empty(4 * hd, dtype=np.float32)
    cdef float[::1] h = np.empty(hd, dtype=np.float32)
    cdef float[::1] c = np.empty(hd, dtype=np.float32)
    _lstm_direction(XGf, WhTf, g, h, c, H, 0, False)
    _lstm_direction(XGb, WhTb, g, h, c, H, hd, True)
    return H_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list np_viterbi(trans, observations):
    # Typed-C reimplementation of the original numpy recursion. The
    # arithmetic is identical: scores accumulate in float64 (the original
    # for_expr was float64 and upcast the float32 trans/observation rows)
    # and ties keep the lowest tag id, like np.argmax.
    cdef:
        int T = len(observations)
        int t, j, k, cur
        double best, val
        double fe[6]
        double vv[6]
        list best_path

    if T == 0:
        return []

    cdef double[:, ::1] tr = np.ascontiguousarray(trans, dtype=np.float64)
    cdef double[:, ::1] ob = np.ascontiguousarray(observations, dtype=np.float64)
    cdef long long[:, ::1] bp = np.empty((T, 6), dtype=np.int64)

    for j in range(6):
        fe[j] = -1e10
    fe[4] = 0 # sp_s = 4

    for t in range(T):
        for j in range(6):
            best = fe[0] + tr[j, 0]
            cur = 0
            for k in range(1, 6):
                val = fe[k] + tr[j, k]
                if val > best:
                    best = val
                    cur = k
            vv[j] = best
            bp[t, j] = cur
        for j in range(6):
            fe[j] = vv[j] + ob[t, j]

    best = fe[0] + tr[5, 0] # sp_e = 5
    cur = 0
    for k in range(1, 6):
        val = fe[k] + tr[5, k]
        if val > best:
            best = val
            cur = k

    best_path = [0]*T
    for t in range(T-1, -1, -1):
        best_path[t] = cur
        cur = bp[t, cur]
    return best_path


cpdef load_file(filename, delimiter='\t', newline='EOS'):
    cdef:
        list X, Y, words, tags
        unicode word, tag

    X = []
    Y = []
    words = []
    tags = []

    with codecs.open(filename, 'r', encoding='utf_8_sig') as f:
        for line in f:
            line = utf8rstrip(line)

            if line == newline:
                if not len(words) == len(tags):
                    raise AssertionError("len(words) != len(tags)")

                X.append(words)
                Y.append(tags)
                words = []
                tags = []
            else:
                line = line.split(delimiter)
                word = " ".join(line[:-1])
                tag = line[-1]
                words.append(word)
                tags.append(tag)

    return X, Y
