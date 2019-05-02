import nagisa

if __name__ == "__main__":
    # Build the tagger by loading the trained model files.
    ud_tagger = nagisa.Tagger(vocabs='ja_gsd_ud.vocabs',
                              params='ja_gsd_ud.params',
                              hp='ja_gsd_ud.hp')

    text = "福岡・博多の観光情報"
    words = ud_tagger.tagging(text)
    print(words)
    #> 福岡/PROPN ・/SYM 博多/PROPN の/ADP 観光/NOUN 情報/NOUN
