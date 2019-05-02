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


if __name__ == "__main__":

    fn_out_test = "ja_gsd_ud.test"
    test_X, test_Y = load_file(fn_out_test)

    ud_tagger = nagisa.Tagger(vocabs='ja_gsd_ud.vocabs',
                              params='ja_gsd_ud.params',
                              hp='ja_gsd_ud.hp')


    true_cm = []
    pred_cm = []
    label2id = {}

    for i in range(len(test_X)):
        words = test_X[i]
        true_tags = test_Y[i]

        # decoding
        pred_tags = ud_tagger.decode(words)

        if true_tags != pred_tags:
            for word, true_tag, pred_tag in zip(words, true_tags, pred_tags):
                if true_tag != pred_tag:

                    if true_tag not in label2id:
                        label2id[true_tag] = len(label2id)

                    if pred_tag not in label2id:
                        label2id[pred_tag] = len(label2id)

                    true_cm.append(label2id[true_tag])
                    pred_cm.append(label2id[pred_tag])


    labels = list(label2id.keys())
    cm = confusion_matrix(true_cm, pred_cm)
    cm_labeled = pd.DataFrame(cm, columns=labels, index=labels)
    print(cm_labeled)
