import nagisa

def write_file(fn_in, fn_out):
    with open(fn_in, "r") as f:
        data = []
        words = []
        postags = []
        for line in f:
            line = line.strip()

            if len(line) > 0:
                prefix = line[0]
                if prefix != "#":
                    tokens = line.split("\t")
                    word = tokens[1]
                    postag = tokens[3]
                    words.append(word)
                    postags.append(postag)

            else:
                if (len(words) > 0) and (len(postags) > 0):
                    data.append([words, postags])
                    words = []
                    postags = []

    with open(fn_out, "w") as f:
        for words, postags in data:
            for word, postag in zip(words, postags):
                f.write("\t".join([word, postag])+"\n")
            f.write("EOS\n")


if __name__ == "__main__":
    # files
    fn_in_train = "UD_Japanese-GSD/ja_gsd-ud-train.conllu"
    fn_in_dev = "UD_Japanese-GSD/ja_gsd-ud-dev.conllu"
    fn_in_test = "UD_Japanese-GSD/ja_gsd-ud-test.conllu"

    fn_out_train = "ja_gsd_ud.train"
    fn_out_dev = "ja_gsd_ud.dev"
    fn_out_test = "ja_gsd_ud.test"

    fn_out_model = "ja_gsd_ud"

    # write files for nagisa
    write_file(fn_in_train, fn_out_train)
    write_file(fn_in_dev, fn_out_dev)
    write_file(fn_in_test, fn_out_test)

    # start training
    nagisa.fit(train_file=fn_out_train, dev_file=fn_out_dev,
               test_file=fn_out_test, model_name=fn_out_model)
