import nagisa

from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report


def main():
    # load the testset
    test_X, test_Y = nagisa.utils.load_file("kwdlc.test")

    # build the tagger for kwdlc
    ner_tagger = nagisa.Tagger(vocabs='kwdlc_ner_model.vocabs',
                              params='kwdlc_ner_model.params',
                              hp='kwdlc_ner_model.hp')

    # predict
    true_Y = []
    pred_Y = []
    for words, true_y in zip(test_X, test_Y):
        pred_y= ner_tagger.decode(words)

        _pred_y = []
        _true_y = []
        for word, pred, true in zip(words, pred_y, true_y):
            _pred_y.append(pred)
            _true_y.append(true)
        true_Y.append(_true_y)
        pred_Y.append(_pred_y)

    # evaluate
    accuracy = accuracy_score(true_Y, pred_Y)
    print("accuracy: {}".format(accuracy))
    f1 = f1_score(true_Y, pred_Y)
    print("macro-f1: {}".format(f1))
    report = classification_report(true_Y, pred_Y)
    print(report)


if __name__ == "__main__":
    main()

