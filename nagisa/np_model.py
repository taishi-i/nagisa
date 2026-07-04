# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np

import nagisa_utils as utils
from nagisa.dynet_loader import load_dynet_text_model, map_nagisa_params

_FORGET_BIAS = 1.0


def _ids_matrix(ids_list):
    return np.asarray(ids_list, dtype=np.intp)


class _LSTM(object):
    """Parameters of one direction of a DyNet VanillaLSTMBuilder, laid out
    for the inference kernels: WxT (in, 4h), WhT (h, 4h) and b (4h,) with
    the +1.0 forget-gate bias folded in."""

    def __init__(self, p):
        self.h_dim = p['Wh'].shape[1]
        self.WxT = np.ascontiguousarray(p['Wx'].T)
        self.WhT = np.ascontiguousarray(p['Wh'].T)
        b = np.array(p['b'], dtype=np.float32)
        b[self.h_dim:2 * self.h_dim] += np.float32(_FORGET_BIAS)
        self.b = b


class _BiRNN(object):
    """dy.BiRNNBuilder.transduce: per layer, forward over X and backward
    over reversed X, outputs concatenated position-wise. Each layer runs
    as one GIL-free fused kernel (nagisa_utils.birnn_transduce)."""

    def __init__(self, lstm_pairs):
        self.layers = [(_LSTM(f), _LSTM(b)) for f, b in lstm_pairs]

    def transduce(self, X):
        for fwd, bwd in self.layers:
            X = utils.birnn_transduce(X, fwd.WxT, fwd.b, fwd.WhT,
                                      bwd.WxT, bwd.b, bwd.WhT)
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
        if len(uids) == 0:
            return []
        vec = utils.build_ws_input(uids, bids, cids, wids_s, wids_e,
                                   self.UNI, self.BI, self.CTYPE, self.WORD)
        hiddens = self.ws_model.transduce(vec)
        observations = utils.linear(hiddens, self.w_wsT, self.b_ws)
        return list(observations)

    def _char_vecs(self, cids_list):
        """vec_char per word: char_seq_model.transduce(chars)[-1].

        The output at the last position concatenates the forward state
        after the full word and the backward state after a single step on
        the last character, so for a 1-layer BiRNN both directions run in
        one GIL-free kernel over the batch of words.
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
        lens = np.array([len(c) for c in unique], dtype=np.intp)
        ids = np.zeros((len(unique), int(lens.max())), dtype=np.intp)
        for n, cids in enumerate(unique):
            ids[n, :len(cids)] = cids

        out = utils.char_vecs_1layer(ids, lens, self.UNI, fwd.WxT, fwd.b,
                                     fwd.WhT, bwd.WxT, bwd.b)
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

        vec = utils.build_pos_input(wids, tids, self.WORD, self.POS,
                                    self.dim_uni)
        vec[:, self.dim_word:self.dim_word + self.dim_uni] = \
            self._char_vecs(cids)
        hiddens = self.pos_model.transduce(vec)
        return utils.linear(hiddens, self.w_posT, self.b_pos)

    def POStagging(self, X):
        logits = self.encode_pt(X)
        return [int(pid) for pid in np.argmax(logits, axis=1)]
