# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np
import dynet_config
dynet_config.set(mem=32, random_seed=1234)
import dynet as dy


class Model(object):

    def __init__(self, hp, params=None, embs=None):
        # Set hyperparameters.
        dim_uni      = hp['DIM_UNI']
        dim_bi       = hp['DIM_BI']
        dim_ctype    = hp['DIM_CTYPE']
        dim_word     = hp['DIM_WORD']
        dim_tag_emb  = hp['DIM_TAGEMB']
        size_uni     = hp['VOCAB_SIZE_UNI']
        size_bi      = hp['VOCAB_SIZE_BI']
        size_word    = hp['VOCAB_SIZE_WORD']
        size_postags = hp['VOCAB_SIZE_POSTAG']
        lr           = hp['LEARNING_RATE']
        ws           = hp['WINDOW_SIZE']
        layers       = hp['LAYERS']
        dim_hidden   = hp['DIM_HIDDEN']
        self.dropout_rate = hp['DROPOUT_RATE']

        # Build a word segmentation model and a POS-tagging model simultaneously.
        model = dy.ParameterCollection()

        # Create embedding matrices.
        # These models share the unigram, bigramn and word embedding matrix.
        self.UNI   = model.add_lookup_parameters((size_uni, dim_uni))
        self.BI    = model.add_lookup_parameters((size_bi, dim_bi))
        if embs is not None:
            self.WORD  = model.add_lookup_parameters((size_word, dim_word), embs)
        else:
            self.WORD  = model.add_lookup_parameters((size_word, dim_word))
        self.CTYPE = model.add_lookup_parameters((7, dim_ctype))
        self.POS   = model.add_lookup_parameters((size_postags, dim_tag_emb))

        # Create a bi-directional LSTM model for word segmentation.
        dim_ws_input  = (dim_uni+dim_bi+dim_ctype)*ws+dim_word*2
        self.ws_model = dy.BiRNNBuilder(layers, dim_ws_input, dim_hidden, model, dy.LSTMBuilder)

        # Create a bi-directional LSTM model for POS-tagging.
        dim_pos_input  = dim_word+dim_uni+dim_tag_emb
        self.pos_model = dy.BiRNNBuilder(layers, dim_pos_input, dim_hidden, model, dy.LSTMBuilder)
        # Create a bi-directional LSTM model for encoding character sequence.
        self.char_seq_model = dy.BiRNNBuilder(layers, dim_uni, dim_uni, model, dy.LSTMBuilder)

        # Create trainer
        self.trainer = dy.SimpleSGDTrainer(model, lr)
        self.trainer.set_clip_threshold(5)

        # Set a tag scheme for WS model.
        dim_output = 4+2
        self.sp_s  = 4
        self.sp_e  = 5

        # Create weight matrices and bias vectors.
        self.dim_output = dim_output
        self.w_ws  = model.add_parameters((dim_output, dim_hidden))
        self.b_ws  = model.add_parameters(dim_output)
        self.trans = model.add_lookup_parameters((dim_output, dim_output))
        self.w_pos = model.add_parameters((size_postags, dim_hidden))
        self.b_pos = model.add_parameters(size_postags)
        self.dim_word    = dim_word
        self.dim_tag_emb = dim_tag_emb

        # Load trained parameters.
        if params:
            model.populate(params)
        self.model = model

        # As nparray
        self.trans_array = self.trans.as_array()


    def encode_ws(self, X, train=False):
        dy.renew_cg()

        # Remove dy.parameters(...) for DyNet v.2.1
        #w_ws = dy.parameter(self.w_ws)
        #b_ws = dy.parameter(self.b_ws)
        w_ws = self.w_ws
        b_ws = self.b_ws

        ipts = []
        length = len(X[0])
        for i in range(length):
            uni   = X[0][i]
            bi    = X[1][i]
            ctype = X[2][i]
            start = X[3][i]
            end   = X[4][i]

            vec_uni   = dy.concatenate([self.UNI[uid] for uid in uni])
            vec_bi    = dy.concatenate([self.BI[bid] for bid in bi])
            vec_start = dy.esum([self.WORD[sid] for sid in start])
            vec_end   = dy.esum([self.WORD[eid] for eid in end])
            vec_ctype = dy.concatenate([self.CTYPE[cid] for cid in ctype])
            vec_at_i  = dy.concatenate([vec_uni, vec_bi, vec_ctype, vec_start, vec_end])

            if train is True:
                vec_at_i = dy.dropout(vec_at_i, self.dropout_rate)
            ipts.append(vec_at_i)

        bilstm_outputs = self.ws_model.transduce(ipts)
        observations   = [w_ws*h+b_ws for h in bilstm_outputs]
        return observations


    def forward(self, observations):

        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dy.pick(scores, argmax_score)
            max_score_expr_broadcast = dy.concatenate([max_score_expr] * self.dim_output)
            return max_score_expr + dy.log(dy.sum_elems(dy.transpose(dy.exp(scores - max_score_expr_broadcast))))

        init_alphas = [-1e10] * self.dim_output
        init_alphas[self.sp_s] = 0
        for_expr = dy.inputVector(init_alphas)
        for obs in observations:
            alphas_t = []
            for next_tag in range(self.dim_output):
                obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * self.dim_output)
                next_tag_expr = for_expr + self.trans[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = dy.concatenate(alphas_t)
        terminal_expr = for_expr + self.trans[self.sp_e]
        alpha = log_sum_exp(terminal_expr)
        return alpha


    def score_sentence(self, observations, tags):
        if not len(observations) == len(tags):
            raise AssertionError("len(observations) != len(tags)")

        score_seq = [0]
        score = dy.scalarInput(0)
        tags = [self.sp_s] + tags
        for i, obs in enumerate(observations):
            score = score + dy.pick(self.trans[tags[i+1]], tags[i]) + dy.pick(obs, tags[i+1])
            score_seq.append(score.value())
        score = score + dy.pick(self.trans[self.sp_e], tags[-1])
        return score


    def viterbi_decoding(self, observations):
        backpointers = []
        init_vvars   = [-1e10] * self.dim_output
        init_vvars[self.sp_s] = 0
        for_expr     = dy.inputVector(init_vvars)
        trans_exprs  = [self.trans[idx] for idx in range(self.dim_output)]
        for obs in observations:
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.dim_output):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id  = np.argmax(next_tag_arr)
                bptrs_t.append(best_tag_id)
                vvars_t.append(dy.pick(next_tag_expr, best_tag_id))
            for_expr = dy.concatenate(vvars_t) + obs
            backpointers.append(bptrs_t)
        terminal_expr = for_expr + trans_exprs[self.sp_e]
        terminal_arr  = terminal_expr.npvalue()
        best_tag_id   = np.argmax(terminal_arr)
        path_score    = dy.pick(terminal_expr, best_tag_id)
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        best_path.reverse()
        if not start == self.sp_s:
            raise AssertionError("start != self.sp_s")
        return best_path, path_score


    def encode_pt(self, X, train=False):
        dy.renew_cg()

        # Remove dy.parameters(...) for DyNet v.2.1
        #w_pos = dy.parameter(self.w_pos)
        #b_pos = dy.parameter(self.b_pos)
        w_pos = self.w_pos
        b_pos = self.b_pos

        ipts  = []
        length = len(X[0])
        for i in range(length):
            cids = X[0][i]
            wid  = X[1][i]
            tids = X[2][i]
            vec_char = self.char_seq_model.transduce([self.UNI[cid] for cid in cids])[-1]

            vec_tags = []
            for tid in tids:
                if tid == 0:
                    zero = dy.inputVector(np.zeros(self.dim_tag_emb))
                    vec_tags.append(zero)
                else:
                    vec_tags.append(self.POS[tid])
            vec_tag = dy.esum(vec_tags)

            if wid == 0:
                vec_word = dy.inputVector(np.zeros(self.dim_word))
            else:
                vec_word = self.WORD[wid]

            vec_at_i = dy.concatenate([vec_word, vec_char, vec_tag])
            if train is True:
                vec_at_i = dy.dropout(vec_at_i, self.dropout_rate)
            ipts.append(vec_at_i)
        hiddens = self.pos_model.transduce(ipts)
        probs = [dy.softmax(w_pos*h+b_pos) for h in hiddens]
        return probs


    def get_POStagging_loss(self, X, Y):
        losses = []
        probs = self.encode_pt(X, train=True)
        for prob, y in zip(probs, Y):
            losses.append(-dy.log(dy.pick(prob, y)))
        loss = dy.esum(losses)
        return loss


    def POStagging(self, X):
        probs = self.encode_pt(X)
        pids = [np.argmax(prob.npvalue()) for prob in probs]
        return pids
