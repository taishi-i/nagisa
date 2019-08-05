import nagisa

import numpy as np
#from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report

import spacy
nlp = spacy.load('ja_ginza')


def main():
    # load the testset
    test_X, test_Y = nagisa.utils.load_file("kwdlc.test")

    # build the tagger for kwdlc
    ner_tagger = nagisa.Tagger(vocabs='kwdlc_ner_model.vocabs',
                              params='kwdlc_ner_model.params',
                              hp='kwdlc_ner_model.hp')

    word2id = ner_tagger._word2id


    W = []
    true_Y = []
    pred_Y = []
    ginza = []
    for words, true_y in zip(test_X, test_Y):

        doc = nlp("".join(words))

        ents = [[ent.text, ent.label_] for ent in doc.ents]
        ginza.append(ents)

        pred_y= ner_tagger.decode(words)

        _W = []
        _pred_y = []
        _true_y = []
        for word, pred, true in zip(words, pred_y, true_y):
            #if word not in word2id:
            #    _W.append(word)
            #    _pred_y.append(pred)
            #    _true_y.append(true)
            _W.append(word)
            _pred_y.append(pred)
            _true_y.append(true)
        W.append(_W)
        true_Y.append(_true_y)
        pred_Y.append(_pred_y)


    #accuracy = accuracy_score(true_Y, pred_Y)
    #print(accuracy)

    f1 = f1_score(true_Y, pred_Y)
    print(f1)

    accuracy = accuracy_score(true_Y, pred_Y)
    print(accuracy)

    report = classification_report(true_Y, pred_Y)
    print(report)

    #for i in range(1):
    #    print(i)
    #    print("word", W[i])
    #    print("true", true_Y[i])
    #    print("pred", pred_Y[i])
    #    print("ginza", ginza[i])

    true_Y = sum(true_Y, [])
    pred_Y = sum(pred_Y, [])

    label2id = {}
    for y in true_Y:
        if y not in label2id:
            label2id[y] = len(label2id)

    id2label = {v:k for k, v in label2id.items()}
    labels = list(label2id.keys())
    cm = confusion_matrix(true_Y, pred_Y, labels=labels)

    label2freq = {}
    for i in range(len(labels)):
        for j in range(len(labels)):
            freq = cm[i][j]
            label_i = id2label[i]
            label_j = id2label[j]
            if label_i != label_j and freq > 0:
                label2freq[" >>> ".join([label_i, label_j])] = freq

    for k, v in sorted(label2freq.items(), key=lambda x:x[1]):
        print(k, v)



if __name__ == "__main__":
    main()
