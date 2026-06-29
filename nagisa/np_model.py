# -*- coding:utf-8 -*-

"""
Numpy implementation of nagisa's neural inference (word segmentation and
POS tagging). Replicates the DyNet computation in nagisa/model.py for
prediction: same architecture, same parameters, same outputs. DyNet is
only needed for training (nagisa.fit).

The LSTM is DyNet's VanillaLSTMBuilder: gates packed as [i; f; o; g] in
the 4*h rows of Wx/Wh/b, with a +1.0 forget-gate bias applied at runtime.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

import nagisa_utils as utils
from nagisa.dynet_loader import load_dynet_text_model, map_nagisa_params

_FORGET_BIAS = 1.0


def _ids_matrix(ids_list):
    return np.asarray(ids_list, dtype=np.intp)


def _ragged_sum(table, ids_list, zero_id=None):
    """Sum embedding rows per group, like [dy.esum([T[i] for i in ids])].

    If zero_id is not None, rows with that id contribute a zero vector
    (mirrors the tid == 0 / wid == 0 special cases in model.py).
    """
    lens = [len(ids) for ids in ids_list]
    if min(lens) == 0:
        # dy.esum([]) raises; keep the same failure mode.
        raise ValueError('esum requires at least one argument')
    flat = []
    for ids in ids_list:
        flat.extend(ids)
    flat = np.asarray(flat, dtype=np.intp)
    vecs = table[flat]
    if zero_id is not None:
        vecs[flat == zero_id] = 0
    if max(lens) == 1:
        return vecs
    offsets = np.zeros(len(lens), dtype=np.intp)
    np.cumsum(lens[:-1], out=offsets[1:])
    return np.add.reduceat(vecs, offsets, axis=0)


class _LSTM(object):
    """One direction of a DyNet VanillaLSTMBuilder, inference only."""

    def __init__(self, p):
        self.h_dim = p['Wh'].shape[1]
        self.WxT = np.ascontiguousarray(p['Wx'].T)
        self.WhT = np.ascontiguousarray(p['Wh'].T)
        b = np.array(p['b'], dtype=np.float32)
        b[self.h_dim:2 * self.h_dim] += np.float32(_FORGET_BIAS)
        self.b = b

    def step_batch(self, XG_t, h, c):
        """One step for a batch of rows: XG_t (N, 4h) includes Wx*x + b.
        Rows are independent, so finished batch lanes do not disturb the
        others."""
        h_dim = self.h_dim
        h3 = 3 * h_dim
        g = XG_t + h.dot(self.WhT)
        ifo = 1.0 / (1.0 + np.exp(-g[:, :h3]))
        u = np.tanh(g[:, h3:])
        c = ifo[:, h_dim:2 * h_dim] * c + ifo[:, :h_dim] * u
        h = ifo[:, 2 * h_dim:] * np.tanh(c)
        return h, c


class _BiRNN(object):
    """dy.BiRNNBuilder.transduce: per layer, forward over X and backward
    over reversed X, outputs concatenated position-wise.

    The input projections Wx*x_t + b run as one BLAS gemm per direction;
    the sequential recurrence runs in Cython (nagisa_utils.birnn_transduce)."""

    def __init__(self, lstm_pairs):
        self.layers = [(_LSTM(f), _LSTM(b)) for f, b in lstm_pairs]

    def transduce(self, X):
        for fwd, bwd in self.layers:
            XGf = X.dot(fwd.WxT)
            XGf += fwd.b
            XGb = X[::-1].dot(bwd.WxT)
            XGb += bwd.b
            X = utils.birnn_transduce(XGf, XGb, fwd.WhT, bwd.WhT)
        return X


class Model(object):
    """Numpy drop-in for the inference surface of nagisa.model.Model:
    encode_ws(), POStagging() and the trans_array attribute."""

    def __init__(self, hp, params=None, embs=None):
        if not params:
            raise ValueError(
                'A trained model file is required for inference. '
                'For training, use nagisa.fit (requires DyNet).')

        self.dim_uni = hp['DIM_UNI']
        self.dim_word = hp['DIM_WORD']
        self.dim_tag_emb = hp['DIM_TAGEMB']
        self.dim_output = 4 + 2

        p = map_nagisa_params(load_dynet_text_model(params), hp)
        self.UNI = p['UNI']
        self.BI = p['BI']
        self.WORD = p['WORD']
        self.CTYPE = p['CTYPE']
        self.POS = p['POS']
        self.w_wsT = np.ascontiguousarray(p['w_ws'].T)
        self.b_ws = p['b_ws']
        self.w_posT = np.ascontiguousarray(p['w_pos'].T)
        self.b_pos = p['b_pos']
        self.ws_model = _BiRNN(p['ws_lstms'])
        self.pos_model = _BiRNN(p['pos_lstms'])
        self.char_seq_model = _BiRNN(p['char_lstms'])
        self.trans_array = p['trans']

    def encode_ws(self, X, train=False):
        if train:
            raise NotImplementedError(
                'Training requires DyNet (see nagisa.model.Model).')
        uids, bids, cids, wids_s, wids_e = X
        T = len(uids)
        if T == 0:
            return []
        vec = np.concatenate([
            self.UNI[_ids_matrix(uids)].reshape(T, -1),
            self.BI[_ids_matrix(bids)].reshape(T, -1),
            self.CTYPE[_ids_matrix(cids)].reshape(T, -1),
            _ragged_sum(self.WORD, wids_s),
            _ragged_sum(self.WORD, wids_e),
        ], axis=1)
        hiddens = self.ws_model.transduce(vec)
        observations = hiddens.dot(self.w_wsT)
        observations += self.b_ws
        return list(observations)

    def _char_vecs(self, cids_list):
        """vec_char per word: char_seq_model.transduce(chars)[-1].

        The output at the last position concatenates the forward state
        after the full word and the backward state after a single step on
        the last character, so for a 1-layer BiRNN the forward pass is
        batched across words and the backward pass needs no recurrence.
        """
        if len(self.char_seq_model.layers) != 1 or min(
                len(c) for c in cids_list) == 0:
            return np.stack([
                self.char_seq_model.transduce(self.UNI[_ids_matrix(c)])[-1]
                for c in cids_list])

        # Identical words get identical vectors; compute unique words only.
        index = {}
        order = []
        unique = []
        for cids in cids_list:
            key = tuple(cids)
            pos = index.get(key)
            if pos is None:
                pos = len(unique)
                index[key] = pos
                unique.append(cids)
            order.append(pos)

        fwd, bwd = self.char_seq_model.layers[0]
        N = len(unique)
        h_dim = fwd.h_dim
        lens = np.array([len(c) for c in unique], dtype=np.intp)
        L_max = int(lens.max())

        ids = np.zeros((N, L_max), dtype=np.intp)
        for n, cids in enumerate(unique):
            ids[n, :len(cids)] = cids

        XG = self.UNI[ids].dot(fwd.WxT)
        XG += fwd.b
        h = np.zeros((N, h_dim), dtype=np.float32)
        c = np.zeros((N, h_dim), dtype=np.float32)
        out = np.empty((N, 2 * h_dim), dtype=np.float32)
        out_fwd = out[:, :h_dim]
        for t in range(L_max):
            h, c = fwd.step_batch(XG[:, t, :], h, c)
            done = np.nonzero(lens == t + 1)[0]
            if done.size:
                out_fwd[done] = h[done]

        # Backward: one step from the zero state on each last character.
        g = self.UNI[ids[np.arange(N), lens - 1]].dot(bwd.WxT)
        g += bwd.b
        ifo = 1.0 / (1.0 + np.exp(-g[:, :3 * h_dim]))
        u = np.tanh(g[:, 3 * h_dim:])
        out[:, h_dim:] = ifo[:, 2 * h_dim:] * np.tanh(ifo[:, :h_dim] * u)
        return out[order]

    def encode_pt(self, X, train=False):
        """Returns POS logits (N, size_postags); softmax is omitted since
        it does not change the argmax."""
        if train:
            raise NotImplementedError(
                'Training requires DyNet (see nagisa.model.Model).')
        cids, wids, tids = X[0], X[1], X[2]
        N = len(cids)
        if N == 0:
            return np.zeros((0, self.b_pos.shape[0]), dtype=np.float32)

        wids_arr = _ids_matrix(wids)
        vec_word = self.WORD[wids_arr].copy()
        vec_word[wids_arr == 0] = 0
        vec = np.concatenate([
            vec_word,
            self._char_vecs(cids),
            _ragged_sum(self.POS, tids, zero_id=0),
        ], axis=1)
        hiddens = self.pos_model.transduce(vec)
        logits = hiddens.dot(self.w_posT)
        logits += self.b_pos
        return logits

    def POStagging(self, X):
        logits = self.encode_pt(X)
        return [int(pid) for pid in np.argmax(logits, axis=1)]
