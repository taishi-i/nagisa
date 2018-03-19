# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import

import re
import sys
import gzip
import numpy as np
import unicodedata

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
            words.append(character)
        elif tag == 2:
            partical_word += character
            words.append(partical_word)
            partical_word = u''
        else:
            partical_word += character
    return words


cpdef dump_data(data, fn):
    with gzip.open(fn, 'wb') as gf:
        cPickle.dump(data, gf, protocol=2)
        gf.close()


cpdef load_data(fn):
    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)


cpdef list np_viterbi(trans, observations):
    cdef:
        int idx, best_tag_id
        list bptrs_t, vvars_t, backpointer, indice, best_path

    for_expr = np.array([-1e10]*6)
    for_expr[4] = 0 # sp_s = 4
    indice = [0,1,2,3,4,5]
    backpointer = []

    for obs in observations:
        bptrs_t = []
        vvars_t = []
        for idx in indice:
            next_tag_expr = for_expr+trans[idx]
            best_tag_id = np.argmax(next_tag_expr)
            bptrs_t.append(best_tag_id)
            vvars_t.append(next_tag_expr[best_tag_id])
        for_expr = np.array(vvars_t) + obs
        backpointer.append(bptrs_t)

    terminal_expr = for_expr + trans[5] # sp_e = 5
    best_tag_id = np.argmax(terminal_expr)
    best_path = [best_tag_id]

    for bptrs_t in reversed(backpointer):
        best_tag_id = bptrs_t[best_tag_id]
        best_path.append(best_tag_id)

    best_path.pop()
    best_path.reverse()
    return best_path
