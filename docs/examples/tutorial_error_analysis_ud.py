import nagisa
import pandas as pd

from sklearn.metrics import confusion_matrix


def load_file(filename):
    X = []
    Y = []
    words = []
    tags = []
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip()
            if line == "EOS":
                assert(len(words) == len(tags))
                X.append(words)
                Y.append(tags)
                words = []
                tags = []
            else:
                line = line.split("\t")
                word = " ".join(line[:-1])
                tag = line[-1]
                words.append(word)
                tags.append(tag)
    return X, Y


def create_confusion_matrix(tagger, X, Y):
    true_cm = []
    pred_cm = []
    label2id = {}
    for i in range(len(X)):
        words = X[i]
        true_tags = Y[i]
        pred_tags = tagger.decode(words) # decoding

        if true_tags != pred_tags:
            for true_tag, pred_tag in zip(true_tags, pred_tags):
                if true_tag != pred_tag:
                    if true_tag not in label2id:
                        label2id[true_tag] = len(label2id)

                    if pred_tag not in label2id:
                        label2id[pred_tag] = len(label2id)

                    true_cm.append(label2id[true_tag])
                    pred_cm.append(label2id[pred_tag])

    cm = confusion_matrix(true_cm, pred_cm)
    labels = list(label2id.keys())
    cm_labeled = pd.DataFrame(cm, columns=labels, index=labels)
    return cm_labeled


if __name__ == "__main__":
    # load the testset
    test_X, test_Y = load_file("ja_gsd_ud.test")

    # build the tagger for UD
    ud_tagger = nagisa.Tagger(vocabs='ja_gsd_ud.vocabs',
                              params='ja_gsd_ud.params',
                              hp='ja_gsd_ud.hp')

    # create a confusion matrix if tagger make a mistake in prediction.
    cm_labeled = create_confusion_matrix(ud_tagger, test_X, test_Y)
    print(cm_labeled)
