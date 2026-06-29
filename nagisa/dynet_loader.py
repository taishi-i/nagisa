# -*- coding:utf-8 -*-

"""
Load DyNet text-format model files with numpy (no DyNet required).

DyNet's TextFileSaver writes one section per parameter:

    #Parameter# /birnn/vanilla-lstm-builder/_0 {200,200} 640001 ZERO_GRAD
    <640001 bytes of space-separated "%+.8e" floats, newline-terminated>

The byte count in the header is the exact length of the value block, and
the "%+.8e" formatting round-trips float32 values exactly. Matrices are
flattened in column-major (Eigen) order.
"""

from __future__ import division, print_function, absolute_import

import re
from collections import OrderedDict

import numpy as np

_HEADER_RE = re.compile(r'^#(Parameter|LookupParameter)# (\S+) \{(\d+(?:,\d+)*)\} (\d+) (\S+)\s*$')


def _parse_values(data):
    text = data.decode('latin-1')
    try:
        return np.fromstring(text, dtype=np.float64, sep=' ')
    except (AttributeError, ValueError):
        # np.fromstring's text mode may go away in future numpy releases.
        return np.array(text.split(), dtype=np.float64)


def load_dynet_text_model(path):
    """Parse a DyNet text-format model file.

    return:
        - OrderedDict mapping parameter name to (kind, array) where kind is
          "Parameter" or "LookupParameter". Plain parameters keep DyNet's
          dim layout (e.g. (6, 100)); lookup parameters are returned as
          (num_entries, dim), the same layout as LookupParameters.as_array().
    """
    params = OrderedDict()
    with open(path, 'rb') as f:
        while True:
            header = f.readline()
            if not header:
                break
            match = _HEADER_RE.match(header.decode('latin-1'))
            if match is None:
                raise ValueError(
                    'Not a DyNet text-format model file (unexpected header %r '
                    'in %s)' % (header[:60], path))
            kind = match.group(1)
            name = match.group(2)
            shape = tuple(int(d) for d in match.group(3).split(','))
            nbytes = int(match.group(4))
            grad_status = match.group(5)
            if grad_status != 'ZERO_GRAD':
                raise ValueError(
                    'Unsupported gradient block %r for parameter %s in %s'
                    % (grad_status, name, path))

            values = _parse_values(f.read(nbytes))
            if values.size != int(np.prod(shape)):
                raise ValueError(
                    'Parameter %s in %s: expected %d values, got %d'
                    % (name, path, int(np.prod(shape)), values.size))

            if len(shape) == 1:
                arr = values
            elif len(shape) == 2:
                # Column-major data; this yields shape (shape[1], shape[0]),
                # i.e. row j is column j of the DyNet tensor.
                arr = values.reshape((shape[1], shape[0]))
                if kind == 'Parameter':
                    arr = arr.T
                # LookupParameters are stored as {dim, num_entries}; keeping
                # (num_entries, dim) matches LookupParameters.as_array().
            else:
                raise ValueError(
                    'Parameter %s in %s: unsupported rank %d'
                    % (name, path, len(shape)))
            params[name] = (kind, np.ascontiguousarray(arr, dtype=np.float32))
    return params


def _check_shape(name, arr, expected):
    if arr.shape != tuple(expected):
        raise ValueError(
            'Model parameter %s has shape %s, expected %s. The model file '
            'does not match the given hyperparameters.'
            % (name, arr.shape, tuple(expected)))


def _group_lstm_params(items, prefix):
    """Group /<prefix>/vanilla-lstm-builder*/_k params into per-builder lists."""
    builders = OrderedDict()
    for name, (kind, arr) in items:
        parts = name.split('/')
        if len(parts) != 4 or parts[1] != prefix:
            raise ValueError('Unexpected LSTM parameter name %r' % name)
        builder, param_id = parts[2], parts[3]
        if 'lstm-builder' in builder and not builder.startswith('vanilla-lstm-builder'):
            raise ValueError(
                'Unsupported RNN builder %r: only models saved with DyNet '
                'VanillaLSTMBuilder (as used by nagisa.fit) are supported.'
                % builder)
        builders.setdefault(builder, []).append((param_id, arr))

    lstms = []
    for builder, plist in builders.items():
        if [pid for pid, _ in plist] != ['_0', '_1', '_2']:
            raise ValueError(
                'Unsupported LSTM parameter layout for %r: expected the '
                '3 parameters (Wx, Wh, b) of a VanillaLSTMBuilder, got %s'
                % (builder, [pid for pid, _ in plist]))
        lstms.append({'Wx': plist[0][1], 'Wh': plist[1][1], 'b': plist[2][1]})
    return lstms


def map_nagisa_params(params, hp):
    """Map parsed parameters onto nagisa's model structure, validating shapes.

    The section order is fixed by Model.__init__ (nagisa/model.py): plain
    parameters appear as the three BiRNNs (ws, pos, char; each with
    2*LAYERS builders of Wx/Wh/b) followed by w_ws, b_ws, w_pos, b_pos;
    lookup parameters appear as UNI, BI, WORD, CTYPE, POS, trans.

    return:
        - dict with keys: UNI, BI, WORD, CTYPE, POS, trans, w_ws, b_ws,
          w_pos, b_pos (numpy arrays) and ws_lstms, pos_lstms, char_lstms
          (lists of (fwd, bwd) dicts with Wx/Wh/b per BiRNN layer).
    """
    layers = hp['LAYERS']
    dim_uni = hp['DIM_UNI']
    dim_hidden = hp['DIM_HIDDEN']
    size_postags = hp['VOCAB_SIZE_POSTAG']
    dim_ws_input = ((dim_uni + hp['DIM_BI'] + hp['DIM_CTYPE']) * hp['WINDOW_SIZE']
                    + 2 * hp['DIM_WORD'])
    dim_pos_input = hp['DIM_WORD'] + dim_uni + hp['DIM_TAGEMB']

    birnn_items = OrderedDict()
    plain = []
    lookups = []
    for name, (kind, arr) in params.items():
        parts = name.split('/')
        if kind == 'Parameter' and len(parts) > 2:
            birnn_items.setdefault(parts[1], []).append((name, (kind, arr)))
        elif kind == 'Parameter':
            plain.append((name, arr))
        else:
            lookups.append((name, arr))

    if len(birnn_items) != 3:
        raise ValueError(
            'Expected 3 BiLSTM parameter groups (ws, pos, char), found %d: %s'
            % (len(birnn_items), list(birnn_items.keys())))
    if len(plain) != 4:
        raise ValueError(
            'Expected 4 plain parameters (w_ws, b_ws, w_pos, b_pos), found %d'
            % len(plain))
    if len(lookups) != 6:
        raise ValueError(
            'Expected 6 lookup parameters (UNI, BI, WORD, CTYPE, POS, trans), '
            'found %d' % len(lookups))

    model = {}
    birnn_specs = [('ws_lstms', dim_ws_input, dim_hidden),
                   ('pos_lstms', dim_pos_input, dim_hidden),
                   ('char_lstms', dim_uni, dim_uni)]
    for (key, dim_input, dim_out), items in zip(birnn_specs, birnn_items.values()):
        lstms = _group_lstm_params(items, items[0][0].split('/')[1])
        if len(lstms) != 2 * layers:
            raise ValueError(
                'Expected %d LSTMs (2 per layer) for %s, found %d'
                % (2 * layers, key, len(lstms)))
        h = dim_out // 2
        pairs = []
        for layer in range(layers):
            in_dim = dim_input if layer == 0 else dim_out
            fwd, bwd = lstms[2 * layer], lstms[2 * layer + 1]
            for tag, lstm in (('fwd', fwd), ('bwd', bwd)):
                base = '%s[%d].%s' % (key, layer, tag)
                _check_shape(base + '.Wx', lstm['Wx'], (4 * h, in_dim))
                _check_shape(base + '.Wh', lstm['Wh'], (4 * h, h))
                _check_shape(base + '.b', lstm['b'], (4 * h,))
            pairs.append((fwd, bwd))
        model[key] = pairs

    for (name, arr), (key, shape) in zip(plain, [
            ('w_ws', (6, dim_hidden)), ('b_ws', (6,)),
            ('w_pos', (size_postags, dim_hidden)), ('b_pos', (size_postags,))]):
        _check_shape(key + ' (%s)' % name, arr, shape)
        model[key] = arr

    for (name, arr), (key, shape) in zip(lookups, [
            ('UNI', (hp['VOCAB_SIZE_UNI'], dim_uni)),
            ('BI', (hp['VOCAB_SIZE_BI'], hp['DIM_BI'])),
            ('WORD', (hp['VOCAB_SIZE_WORD'], hp['DIM_WORD'])),
            ('CTYPE', (7, hp['DIM_CTYPE'])),
            ('POS', (size_postags, hp['DIM_TAGEMB'])),
            ('trans', (6, 6))]):
        _check_shape(key + ' (%s)' % name, arr, shape)
        model[key] = arr

    return model
