import nagisa


def main():
    # build the tagger for kwdlc
    ner_tagger = nagisa.Tagger(vocabs='kwdlc_ner_model.vocabs',
                               params='kwdlc_ner_model.params',
                               hp='kwdlc_ner_model.hp')

    # predict
    text = "鈴木花子さんは2020年に京都大学を卒業しました。"
    tokens = ner_tagger.tagging(text)
    print(tokens)
    #> 鈴木/B-PERSON 花子/I-PERSON さん/O は/O 2020/B-DATE 年/I-DATE に/O 京都/B-ORGANIZATION 大学/I-ORGANIZATION を/O 卒業/O し/O ました/O 。/O


if __name__ == "__main__":
    main()
