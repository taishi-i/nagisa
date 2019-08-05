import random

import nagisa


def write_file(filename, X, Y):
    with open(filename, "w") as f:
        for x, y in zip(X, Y):
            for word, tag in zip(x, y):
                f.write("\t".join([word, tag])+"\n")
            f.write("EOS\n")


def main():
    random.seed(1234)

    # preprocess
    fn_in = "kwdlc.txt"
    X, Y = nagisa.utils.load_file(fn_in)
    indice = [i for i in range(len(X))]
    random.shuffle(indice)

    num_train = int(0.8 * len(indice))
    num_dev = int(0.1 * len(indice))
    num_test = int(0.1 * len(indice))

    train_X = [X[i] for i in indice[:num_train]]
    train_Y = [Y[i] for i in indice[:num_train]]
    dev_X = [X[i] for i in indice[num_train:num_train+num_dev]]
    dev_Y = [Y[i] for i in indice[num_train:num_train+num_dev]]
    test_X = [X[i] for i in indice[num_train+num_dev:num_train+num_dev+num_test]]
    test_Y = [Y[i] for i in indice[num_train+num_dev:num_train+num_dev+num_test]]

    fn_out_train = "kwdlc.train"
    fn_out_dev = "kwdlc.dev"
    fn_out_test = "kwdlc.test"
    write_file(fn_out_train, train_X, train_Y)
    write_file(fn_out_dev, dev_X, dev_Y)
    write_file(fn_out_test, test_X, test_Y)

    # start training
    fn_out_model = "kwdlc_ner_model"
    nagisa.fit(
        train_file=fn_out_train,
        dev_file=fn_out_dev,
        test_file=fn_out_test,
        model_name=fn_out_model
    )


if __name__ == "__main__":
    main()


