from __future__ import division, print_function, absolute_import

import re
import sys
import gzip
import numpy as np
import unicodedata

cimport cython
from libc.math cimport expf, tanhf
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from six.moves import cPickle

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


cdef extern from *:
    """
    #include <math.h>
    #include <stddef.h>

    /* MSVC's C mode does not recognize the bare C99 'restrict' keyword
       (only its own '__restrict' spelling); every other supported
       compiler (gcc, clang, apple clang) accepts 'restrict' directly. */
    #if defined(_MSC_VER) && !defined(__clang__)
    #  define restrict __restrict
    #endif

    #if defined(__x86_64__) && defined(__gnu_linux__) && defined(__GNUC__) \\
        && !defined(__clang__) && (__GNUC__ >= 11)
    #  define NAGISA_TARGET_CLONES \\
          __attribute__((target_clones("avx2,fma", "default")))
    #else
    #  define NAGISA_TARGET_CLONES
    #endif

    /* C(M,N) = A(M,K) @ B(K,N) + bias(N); row-major, lda/ldc row strides.
       Each output row accumulates over k in ascending order, so results do
       not depend on the row-blocking below. */
    NAGISA_TARGET_CLONES
    static void nagisa_sgemm_bias(const float* restrict A, ptrdiff_t lda,
                                  const float* restrict B,
                                  const float* restrict bias,
                                  float* restrict C, ptrdiff_t ldc,
                                  ptrdiff_t M, ptrdiff_t K, ptrdiff_t N)
    {
        ptrdiff_t i = 0, j, k;
        for (; i + 4 <= M; i += 4) {
            const float *a0 = A + i*lda, *a1 = a0 + lda;
            const float *a2 = a1 + lda, *a3 = a2 + lda;
            float *c0 = C + i*ldc, *c1 = c0 + ldc, *c2 = c1 + ldc, *c3 = c2 + ldc;
            for (j = 0; j < N; j++) {
                c0[j] = bias[j]; c1[j] = bias[j];
                c2[j] = bias[j]; c3[j] = bias[j];
            }
            for (k = 0; k < K; k++) {
                const float *bk = B + k*N;
                float v0 = a0[k], v1 = a1[k], v2 = a2[k], v3 = a3[k];
                for (j = 0; j < N; j++) {
                    c0[j] += v0 * bk[j];
                    c1[j] += v1 * bk[j];
                    c2[j] += v2 * bk[j];
                    c3[j] += v3 * bk[j];
                }
            }
        }
        for (; i < M; i++) {
            const float *a0 = A + i*lda;
            float *c0 = C + i*ldc;
            for (j = 0; j < N; j++) c0[j] = bias[j];
            for (k = 0; k < K; k++) {
                const float *bk = B + k*N;
                float v0 = a0[k];
                for (j = 0; j < N; j++) c0[j] += v0 * bk[j];
            }
        }
    }

    /* One LSTM direction over precomputed inputs XG[t] = Wx*x_t + b (the
       bias already includes DyNet's +1 forget-gate bias). Gate columns are
       packed [i, f, o, g], DyNet VanillaLSTM semantics. Hidden states are
       written to H[row, col_off:col_off+hd]; the backward direction
       (reverse != 0) walks rows T-1..0 in place. With H == NULL nothing is
       stored and h/c end up holding the final states. */
    NAGISA_TARGET_CLONES
    static void nagisa_lstm_seq(const float* restrict XG,
                                const float* restrict WhT,
                                float* restrict g, float* restrict h,
                                float* restrict c,
                                float* H, ptrdiff_t ldh, ptrdiff_t col_off,
                                ptrdiff_t T, ptrdiff_t hd, int reverse)
    {
        ptrdiff_t h4 = 4*hd, t, row, j, k;
        for (j = 0; j < hd; j++) { h[j] = 0.0f; c[j] = 0.0f; }
        for (t = 0; t < T; t++) {
            row = reverse ? (T - 1 - t) : t;
            {
                const float *xrow = XG + row*h4;
                for (j = 0; j < h4; j++) g[j] = xrow[j];
            }
            for (k = 0; k < hd; k++) {
                float hk = h[k];
                const float *whk = WhT + k*h4;
                for (j = 0; j < h4; j++) g[j] += hk * whk[j];
            }
            {
                float *out = H ? (H + row*ldh + col_off) : (float*)0;
                for (j = 0; j < hd; j++) {
                    float ct = ((1.0 / (1.0 + expf(-g[hd + j]))) * c[j]
                                + (1.0 / (1.0 + expf(-g[j]))) * tanhf(g[3*hd + j]));
                    c[j] = ct;
                    h[j] = (1.0 / (1.0 + expf(-g[2*hd + j]))) * tanhf(ct);
                    if (out) out[j] = h[j];
                }
            }
        }
    }
    """
    void nagisa_sgemm_bias(const float* A, Py_ssize_t lda, const float* B,
                           const float* bias, float* C, Py_ssize_t ldc,
                           Py_ssize_t M, Py_ssize_t K, Py_ssize_t N) nogil
    void nagisa_lstm_seq(const float* XG, const float* WhT, float* g,
                         float* h, float* c, float* H, Py_ssize_t ldh,
                         Py_ssize_t col_off, Py_ssize_t T, Py_ssize_t hd,
                         int reverse) nogil


def birnn_transduce(X, WxTf, bf, WhTf, WxTb, bb, WhTb):
    """One BiLSTM layer, dy.BiRNNBuilder.transduce semantics: the input
    projections Wx*x_t + b for both directions plus the sequential
    recurrences, GIL-free. Returns H (T, 2h) float32 with
    H[t] = [forward state at t, backward state for position t]."""
    cdef float[:, ::1] Xv = X
    cdef float[:, ::1] Wxf = WxTf
    cdef float[::1] bfv = bf
    cdef float[:, ::1] Whf = WhTf
    cdef float[:, ::1] Wxb = WxTb
    cdef float[::1] bbv = bb
    cdef float[:, ::1] Whb = WhTb
    cdef Py_ssize_t T = Xv.shape[0]
    cdef Py_ssize_t D = Xv.shape[1]
    cdef Py_ssize_t h4 = Wxf.shape[1]
    cdef Py_ssize_t hd = h4 // 4

    H_arr = np.empty((T, 2 * hd), dtype=np.float32)
    if T == 0:
        return H_arr
    cdef float[:, ::1] Hv = H_arr
    cdef float* XG
    cdef float* work
    with nogil:
        XG = <float*> malloc(T * h4 * sizeof(float))
        work = <float*> malloc((h4 + 2 * hd) * sizeof(float))
        if XG == NULL or work == NULL:
            free(XG)
            free(work)
            with gil:
                raise MemoryError()
        nagisa_sgemm_bias(&Xv[0, 0], D, &Wxf[0, 0], &bfv[0], XG, h4, T, D, h4)
        nagisa_lstm_seq(XG, &Whf[0, 0], work, work + h4, work + h4 + hd,
                        &Hv[0, 0], 2 * hd, 0, T, hd, 0)
        nagisa_sgemm_bias(&Xv[0, 0], D, &Wxb[0, 0], &bbv[0], XG, h4, T, D, h4)
        nagisa_lstm_seq(XG, &Whb[0, 0], work, work + h4, work + h4 + hd,
                        &Hv[0, 0], 2 * hd, hd, T, hd, 1)
        free(XG)
        free(work)
    return H_arr


def linear(X, WT, b):
    """Y = X @ WT + b for float32 C-contiguous arrays, GIL-free."""
    cdef float[:, ::1] Xv = X
    cdef float[:, ::1] Wv = WT
    cdef float[::1] bv = b
    cdef Py_ssize_t M = Xv.shape[0]
    cdef Py_ssize_t K = Xv.shape[1]
    cdef Py_ssize_t N = Wv.shape[1]
    Y_arr = np.empty((M, N), dtype=np.float32)
    if M == 0:
        return Y_arr
    cdef float[:, ::1] Yv = Y_arr
    with nogil:
        nagisa_sgemm_bias(&Xv[0, 0], K, &Wv[0, 0], &bv[0], &Yv[0, 0], N, M, K, N)
    return Y_arr


@cython.boundscheck(False)
@cython.wraparound(False)
def char_vecs_1layer(ids, lens, UNI, WxTf, bf, WhTf, WxTb, bb):
    """vec_char for a batch of words through the 1-layer char BiLSTM,
    equal to char_seq_model.transduce(chars)[-1] per word: the forward
    state after the whole word next to the backward state after a single
    step on the last character. ids is (N, L_max) intp with row n padded
    beyond lens[n]; every lens[n] must be >= 1. GIL-free."""
    cdef Py_ssize_t[:, ::1] idv = ids
    cdef Py_ssize_t[::1] lnv = lens
    cdef float[:, ::1] E = UNI
    cdef float[:, ::1] Wxf = WxTf
    cdef float[::1] bfv = bf
    cdef float[:, ::1] Whf = WhTf
    cdef float[:, ::1] Wxb = WxTb
    cdef float[::1] bbv = bb
    cdef Py_ssize_t N = idv.shape[0]
    cdef Py_ssize_t Lmax = idv.shape[1]
    cdef Py_ssize_t du = E.shape[1]
    cdef Py_ssize_t h4 = Wxf.shape[1]
    cdef Py_ssize_t hd = h4 // 4

    out_arr = np.empty((N, 2 * hd), dtype=np.float32)
    if N == 0:
        return out_arr
    # Match numpy fancy indexing: reject ids outside the embedding table
    # instead of reading out of bounds (e.g. mismatched vocab/model files).
    if Lmax and (int(ids.min()) < 0 or int(ids.max()) >= E.shape[0]):
        raise IndexError('character id out of bounds for the embedding '
                         'table with size %d' % E.shape[0])
    if int(lens.max()) > Lmax:
        raise ValueError('lens exceed the padded ids width')
    cdef float[:, ::1] out = out_arr
    cdef float* XG
    cdef float* work
    cdef float* g
    cdef float* h
    cdef float* c
    cdef Py_ssize_t n, t, j, L
    cdef float ct
    with nogil:
        XG = <float*> malloc(Lmax * h4 * sizeof(float))
        work = <float*> malloc((h4 + 2 * hd) * sizeof(float))
        if XG == NULL or work == NULL:
            free(XG)
            free(work)
            with gil:
                raise MemoryError()
        g = work
        h = work + h4
        c = work + h4 + hd
        for n in range(N):
            L = lnv[n]
            if L <= 0:
                for j in range(2 * hd):
                    out[n, j] = 0
                continue
            # forward direction: project this word's characters, then recur
            for t in range(L):
                nagisa_sgemm_bias(&E[idv[n, t], 0], du, &Wxf[0, 0], &bfv[0],
                                  XG + t * h4, h4, 1, du, h4)
            nagisa_lstm_seq(XG, &Whf[0, 0], g, h, c, NULL, 0, 0, L, hd, 0)
            for j in range(hd):
                out[n, j] = h[j]
            # backward direction: one step from the zero state on the last
            # character, so f*c drops out and only the i/o/g gates remain.
            nagisa_sgemm_bias(&E[idv[n, L - 1], 0], du, &Wxb[0, 0], &bbv[0],
                              g, h4, 1, du, h4)
            for j in range(hd):
                ct = (1.0 / (1.0 + expf(-g[j]))) * tanhf(g[3 * hd + j])
                out[n, hd + j] = (1.0 / (1.0 + expf(-g[2 * hd + j]))) * tanhf(ct)
        free(XG)
        free(work)
    return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
def build_ws_input(list uids, list bids, list cids, list wids_s, list wids_e,
                   UNI, BI, CTYPE, WORD):
    """Input matrix of the word-segmentation BiLSTM: per character position
    the concatenation [unigram window embs | bigram window embs | chartype
    window embs | sum of word embs starting here | sum of word embs ending
    here]. One pass instead of a chain of numpy gathers/reduceat/concatenate."""
    cdef float[:, ::1] Ue = UNI
    cdef float[:, ::1] Be = BI
    cdef float[:, ::1] Ce = CTYPE
    cdef float[:, ::1] We = WORD
    cdef Py_ssize_t T = len(uids)
    cdef Py_ssize_t du = Ue.shape[1]
    cdef Py_ssize_t db = Be.shape[1]
    cdef Py_ssize_t dc = Ce.shape[1]
    cdef Py_ssize_t dw = We.shape[1]
    cdef Py_ssize_t W = 0
    if T > 0:
        W = len(<list> uids[0])
    cdef Py_ssize_t off_b = W * du
    cdef Py_ssize_t off_c = W * (du + db)
    cdef Py_ssize_t off_s = W * (du + db + dc)
    cdef Py_ssize_t off_e = off_s + dw
    cdef Py_ssize_t D = off_e + dw

    X_arr = np.empty((T, D), dtype=np.float32)
    cdef float[:, ::1] X = X_arr
    cdef Py_ssize_t nU = Ue.shape[0]
    cdef Py_ssize_t nB = Be.shape[0]
    cdef Py_ssize_t nC = Ce.shape[0]
    cdef Py_ssize_t nW = We.shape[0]
    cdef Py_ssize_t t, w, j, k, idx, n_ids
    cdef list row
    cdef float* dst

    for t in range(T):
        row = <list> uids[t]
        for w in range(W):
            idx = <Py_ssize_t> row[w]
            if idx < 0 or idx >= nU:
                raise IndexError('unigram id %d out of bounds (%d)' % (idx, nU))
            memcpy(&X[t, w * du], &Ue[idx, 0], du * sizeof(float))
        row = <list> bids[t]
        for w in range(W):
            idx = <Py_ssize_t> row[w]
            if idx < 0 or idx >= nB:
                raise IndexError('bigram id %d out of bounds (%d)' % (idx, nB))
            memcpy(&X[t, off_b + w * db], &Be[idx, 0], db * sizeof(float))
        row = <list> cids[t]
        for w in range(W):
            idx = <Py_ssize_t> row[w]
            if idx < 0 or idx >= nC:
                raise IndexError('chartype id %d out of bounds (%d)' % (idx, nC))
            memcpy(&X[t, off_c + w * dc], &Ce[idx, 0], dc * sizeof(float))

        row = <list> wids_s[t]
        n_ids = len(row)
        if n_ids == 0:
            raise ValueError('esum requires at least one argument')
        dst = &X[t, off_s]
        for j in range(dw):
            dst[j] = 0
        for k in range(n_ids):
            idx = <Py_ssize_t> row[k]
            if idx < 0 or idx >= nW:
                raise IndexError('word id %d out of bounds (%d)' % (idx, nW))
            for j in range(dw):
                dst[j] += We[idx, j]

        row = <list> wids_e[t]
        n_ids = len(row)
        if n_ids == 0:
            raise ValueError('esum requires at least one argument')
        dst = &X[t, off_e]
        for j in range(dw):
            dst[j] = 0
        for k in range(n_ids):
            idx = <Py_ssize_t> row[k]
            if idx < 0 or idx >= nW:
                raise IndexError('word id %d out of bounds (%d)' % (idx, nW))
            for j in range(dw):
                dst[j] += We[idx, j]
    return X_arr


@cython.boundscheck(False)
@cython.wraparound(False)
def build_pos_input(list wids, list tids, WORD, POS, Py_ssize_t dim_mid):
    """Input rows of the POS-tagging BiLSTM: [word emb (zeros for id 0) |
    dim_mid gap the caller fills with the char BiLSTM vector | sum of
    POS-tag embs where id 0 contributes a zero vector]."""
    cdef float[:, ::1] We = WORD
    cdef float[:, ::1] Pe = POS
    cdef Py_ssize_t N = len(wids)
    cdef Py_ssize_t dw = We.shape[1]
    cdef Py_ssize_t dp = Pe.shape[1]
    cdef Py_ssize_t off_p = dw + dim_mid

    X_arr = np.empty((N, off_p + dp), dtype=np.float32)
    cdef float[:, ::1] X = X_arr
    cdef Py_ssize_t nW = We.shape[0]
    cdef Py_ssize_t nP = Pe.shape[0]
    cdef Py_ssize_t n, j, k, idx, n_ids
    cdef list row
    cdef float* dst

    for n in range(N):
        idx = <Py_ssize_t> wids[n]
        dst = &X[n, 0]
        if idx == 0:
            for j in range(dw):
                dst[j] = 0
        else:
            if idx < 0 or idx >= nW:
                raise IndexError('word id %d out of bounds (%d)' % (idx, nW))
            memcpy(dst, &We[idx, 0], dw * sizeof(float))

        row = <list> tids[n]
        n_ids = len(row)
        if n_ids == 0:
            raise ValueError('esum requires at least one argument')
        dst = &X[n, off_p]
        for j in range(dp):
            dst[j] = 0
        for k in range(n_ids):
            idx = <Py_ssize_t> row[k]
            if idx != 0:
                if idx < 0 or idx >= nP:
                    raise IndexError('postag id %d out of bounds (%d)' % (idx, nP))
                for j in range(dp):
                    dst[j] += Pe[idx, j]
    return X_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list np_viterbi(trans, observations):
    # Typed-C reimplementation of the original numpy recursion, GIL-free.
    # The arithmetic is identical: scores accumulate in float64 (the
    # original for_expr was float64 and upcast the float32
    # trans/observation rows) and ties keep the lowest tag id, like
    # np.argmax.
    cdef Py_ssize_t T = len(observations)
    if T == 0:
        return []

    cdef double[:, ::1] tr = np.ascontiguousarray(trans, dtype=np.float64)
    cdef double[:, ::1] ob = np.ascontiguousarray(observations, dtype=np.float64)
    bp_arr = np.empty((T, 6), dtype=np.int64)
    path_arr = np.empty(T, dtype=np.int64)
    cdef long long[:, ::1] bp = bp_arr
    cdef long long[::1] path = path_arr
    cdef Py_ssize_t t, j, k, cur
    cdef double best, val
    cdef double fe[6]
    cdef double vv[6]

    with nogil:
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

        for t in range(T-1, -1, -1):
            path[t] = cur
            cur = <Py_ssize_t> bp[t, cur]
    return path_arr.tolist()


cpdef load_file(filename, delimiter='\t', newline='EOS'):
    cdef:
        list X, Y, words, tags
        unicode word, tag

    X = []
    Y = []
    words = []
    tags = []

    with open(filename, 'r', encoding='utf_8_sig') as f:
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
