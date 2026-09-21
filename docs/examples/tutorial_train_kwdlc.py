import nagisa


def main():
    fn_in_train = "kwdlc.train"
    fn_in_dev = "kwdlc.dev"
    fn_in_test = "kwdlc.test"

    # start training
    fn_out_model = "kwdlc_ner_model"
    nagisa.fit(
        train_file=fn_in_train,
        dev_file=fn_in_dev,
        test_file=fn_in_test,
        model_name=fn_out_model
    )


if __name__ == "__main__":
    main()
