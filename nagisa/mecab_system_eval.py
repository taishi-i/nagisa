# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import

import sys
import codecs

if sys.version_info.major == 3:
    PY_3 = True
else:
    PY_3 = False


def readFile(filename):
    sent = []
    data = []
    with codecs.open(filename, 'r', encoding='utf_8_sig') as f:
        for line in f:
            line = line.rstrip()
            if line == "EOS":
                data.append(sent)
                sent = []
            else:
                surface, csv_form = line.split("\t")
                csv_form = line
                if PY_3 is True:
                    surface  = surface.encode("UTF-8")
                    csv_form = csv_form.encode("UTF-8")
                sent.append([surface, csv_form])
    return data


def mecab_eval(sys_data, ans_data):
    """
    This script is written by referring to the following code.
    https://github.com/taku910/mecab/blob/master/mecab/src/eval.cpp
    """
    if not len(sys_data) == len(ans_data):
        raise AssertionError("len(sys_data) != len(ans_data)")
    num_sents = len(sys_data)

    prec = 0
    recall = 0
    num_correct = [0, 0]
    for i in range(num_sents):
        i_sys = 0
        i_ans = 0
        l_sys = 0
        l_ans = 0
        sys_sent = sys_data[i]
        ans_sent = ans_data[i]

        while ((i_sys < len(sys_sent)) & (i_ans < len(ans_sent))):
            if (l_sys == l_ans):
                if (sys_sent[i_sys][0] == ans_sent[i_ans][0]):
                    num_correct[0] += 1 # Word Segmentation

                if (sys_sent[i_sys][1] == ans_sent[i_ans][1]):
                    num_correct[1] += 1 # POS-tagging

                l_sys  += len(sys_sent[i_sys][0])
                l_ans  += len(ans_sent[i_ans][0])
                i_sys  += 1
                i_ans  += 1
                prec   += 1
                recall += 1
            elif (l_sys < l_ans):
                l_sys  += len(sys_sent[i_sys][0])
                i_sys  += 1
                prec   += 1
            else:
                l_ans  += len(ans_sent[i_ans][0])
                i_ans  += 1
                recall += 1

        while (i_sys < len(sys_sent)):
            i_sys  += 1
            prec   += 1

        while (i_ans < len(ans_sent)):
            i_ans  += 1
            recall += 1

    ws_c = num_correct[0]
    pt_c = num_correct[1]
    return [ws_c, pt_c, prec, recall]


def calculate_fvalues(r):
    ws_c, pt_c, prec, recall = r

    ws_p = round(100*ws_c/prec, 4)
    ws_r = round(100*ws_c/recall, 4)
    if ws_p+ws_r == 0:
        ws_f = round(0., 4)
    else:
        ws_f = round(2*ws_p*ws_r/(ws_p+ws_r), 4)

    pt_p = round(100*pt_c/prec, 4)
    pt_r = round(100*pt_c/recall, 4)
    if pt_p+pt_r == 0:
        pt_f = round(0., 4)
    else:
        pt_f = round(2*pt_p*pt_r/(pt_p+pt_r), 4)

    return [ws_p, ws_r, ws_f, pt_p, pt_r, pt_f]


def print_eval(r):
    ws_c, pt_c, prec, recall = r
    ws_p, ws_r, ws_f, pt_p, pt_r, pt_f = calculate_fvalues(r)

    ws_p_out = str(ws_p)+"("+str(ws_c)+"/"+str(prec)+")"
    ws_r_out = str(ws_r)+"("+str(ws_c)+"/"+str(recall)+")"
    pt_p_out = str(pt_p)+"("+str(pt_c)+"/"+str(prec)+")"
    pt_r_out = str(pt_r)+"("+str(pt_c)+"/"+str(recall)+")"

    ws_out = ["LEVEL 0:", ws_p_out, ws_r_out, str(ws_f)]
    pt_out = ["LEVEL ALL:", pt_p_out, pt_r_out, str(pt_f)]

    header = ["        ", "precision", "recall", "F"]
    print("\t".join(header))
    print("\t".join(ws_out))
    print("\t".join(pt_out))


if __name__ == "__main__":
    import argparse

    dsc = "Reimplementation of mecab-system-eval \
           (https://github.com/taku910/mecab/blob/master/mecab/src/eval.cpp). \
           However, this script only evaluates on the word segmentation level \
           and POS-tagging level. Input file format: Word\tPOS-tag"

    parser = argparse.ArgumentParser(description=dsc)
    parser.add_argument("--system", type=str, help="System output's file")
    parser.add_argument("--answer", type=str, help="Answer file")
    args = parser.parse_args()

    sys_data = readFile(args.system)
    ans_data = readFile(args.answer)

    r = mecab_eval(sys_data, ans_data)
    print_eval(r)
