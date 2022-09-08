# -*- coding:utf-8 -*-

import unittest

import nagisa


class TestNagisa(unittest.TestCase):
    def test_tagging(self):
        # test_1
        text   = 'Pythonで簡単に使えるツールです'
        output = 'Python/名詞 で/助詞 簡単/形状詞 に/助動詞 使える/動詞 ツール/名詞 です/助動詞'
        words = nagisa.tagging(text)
        self.assertEqual(output, str(words))

        # test_2
        output = 'python/名詞 で/助詞 簡単/形状詞 に/助動詞 使える/動詞 ツール/名詞 です/助動詞'
        words = nagisa.tagging(text, lower=True)
        self.assertEqual(output, str(words))

        # test_3
        text   = 'ニューラルネットワークを使ってます。'
        output = 'ニューラル/名詞 ネットワーク/名詞 を/助詞 使っ/動詞 て/助動詞 ます/助動詞 。/補助記号'
        self.assertEqual(output, str(nagisa.tagging(text)))

        # test_4
        tagger_nn = nagisa.Tagger(single_word_list=['ニューラルネットワーク', "ニューラルネット"])
        output = 'ニューラルネットワーク/名詞 を/助詞 使っ/動詞 て/助動詞 ます/助動詞 。/補助記号'
        self.assertEqual(output, str(tagger_nn.tagging(text)))

        # test_5
        text = "3月に見た「3月のライオン」"
        new_tagger = nagisa.Tagger(single_word_list=['3月のライオン'])
        output = '3/名詞 月/名詞 に/助詞 見/動詞 た/助動詞 「/補助記号 3月のライオン/名詞 」/補助記号'
        self.assertEqual(output, str(new_tagger.tagging(text)))

        # test_6
        text = "それが、iPhone XSです。"
        output = "それ/代名詞 が/助詞 、/補助記号 iPhone　XS/名詞 です/助動詞 。/補助記号"
        new_tagger = nagisa.Tagger(single_word_list=["iPhone[a-zA-Z0-9 ]+"])

        self.assertEqual(output, str(new_tagger.tagging(text)))

        # test_7
        text = "1234abc ABC"
        output = "1234/名詞 abc　ABC/名詞"
        new_tagger = nagisa.Tagger(single_word_list=["[a-zA-Z ]+", "[0-9]+"])

        self.assertEqual(output, str(new_tagger.tagging(text)))

        # test_8
        text   = '(人•ᴗ•♡)こんばんは♪'
        output = '(人•ᴗ•♡)/補助記号 こんばんは/感動詞 ♪/補助記号'
        words  = nagisa.tagging(text)
        self.assertEqual(output, str(words))

        # test_9
        url    = 'https://github.com/taishi-i/nagisaでコードを公開中(๑¯ω¯๑)'
        output = 'コード/名詞 公開/名詞 中/接尾辞'
        words  = nagisa.filter(url, filter_postags=['URL', '補助記号', '助詞'])
        self.assertEqual(output, str(words))

        # test_10
        output = 'https://github.com/taishi-i/nagisa/URL で/助詞 を/助詞 (๑　̄ω　̄๑)/補助記号'
        words  = nagisa.extract(url, extract_postags=['URL', '補助記号', '助詞'])
        self.assertEqual(output, str(words))

        # test_11
        words  = [" (人•ᴗ•♡)","こんばんは","♪"]
        output = ['補助記号', '感動詞', '補助記号']
        postags = nagisa.postagging(words)
        self.assertEqual(output, postags)

        # test_12
        postags = nagisa.decode(words)
        self.assertEqual(output, postags)

        # test_13
        words  = [" (人•ᴗ•♡)", "　", "こんばんは","♪"]
        output = ['補助記号', "空白", '感動詞', '補助記号']
        postags = nagisa.postagging(words)

        self.assertEqual(output, postags)

        # test_14
        postags = nagisa.decode(words)
        self.assertEqual(output, postags)

        # test_15
        words = [" (人•ᴗ•♡)", " ", "こんばんは","♪"]
        output = ['補助記号', "空白", '感動詞', '補助記号']
        postags = nagisa.postagging(words)

        self.assertEqual(output, postags)

        # test_16
        postags = nagisa.decode(words)
        self.assertEqual(output, postags)

        # test_17
        text = "こんばんは😀"
        output = "こんばんは/感動詞 😀/補助記号"
        words = nagisa.tagging(text)
        self.assertEqual(output, str(words))

        # test_18
        text = "ｺﾝﾊﾞﾝﾊ１２３４５"
        output = "コンバンハ/名詞 1/名詞 2/名詞 3/名詞 4/名詞 5/名詞"
        words = nagisa.tagging(text)
        self.assertEqual(output, str(words))

        # test_19
        text = "𪗱𪘂𪘚𪚲"
        output = "𪗱/補助記号 𪘂/補助記号 𪘚/補助記号 𪚲/補助記号"
        words = nagisa.tagging(text)
        self.assertEqual(output, str(words))

        # test_26
        text = "エラーを避けるため、İはIに変換される"
        output = "エラー/名詞 を/助詞 避ける/動詞 ため/名詞 、/補助記号 I/名詞 は/助詞 I/名詞 に/助詞 変換/名詞 さ/動詞 れる/助動詞"
        words = nagisa.tagging(text)
        self.assertEqual(output, str(words))


    def test_utils(self):
        # test_20
        output = "oov"
        self.assertEqual(output, nagisa.utils.OOV)

        # test_21
        output = "pad"
        self.assertEqual(output, nagisa.utils.PAD)

        # test_22
        text = "Ｐｙｔｈｏｎ"
        text = nagisa.utils.preprocess(text)
        text = text.lower()
        output = ['p', 'y', 't', 'h', 'o', 'n']
        unigrams = nagisa.utils.get_unigram(text)

        self.assertEqual(output, unigrams)

    def test_fit(self):
        # test_22
        nagisa.fit(
            train_file="nagisa/data/sample_datasets/sample.train",
            dev_file="nagisa/data/sample_datasets/sample.dev",
            test_file="nagisa/data/sample_datasets/sample.test",
            model_name="sample",
        )

        # test_23
        nagisa.fit(
            train_file="nagisa/data/sample_datasets/sample.train",
            dev_file="nagisa/data/sample_datasets/sample.dev",
            test_file="nagisa/data/sample_datasets/sample.test",
            dict_file="nagisa/data/sample_datasets/sample.dict",
            emb_file="nagisa/data/sample_datasets/sample.emb",
            model_name="sample",
            newline="EOS",
            delimiter="\t"
        )

        # test_24
        nagisa.fit(
            train_file="nagisa/data/sample_datasets/sample.train",
            dev_file="nagisa/data/sample_datasets/sample.dev",
            test_file="nagisa/data/sample_datasets/sample.test",
            dict_file="nagisa/data/sample_datasets/sample.dict",
            emb_file="nagisa/data/sample_datasets/sample.emb",
            model_name="sample",
            newline="EOS",
            delimiter="\t",
            min_count=0
        )

    def test_mecab_system_eval(self):
        # test_25
        system_file = "nagisa/data/sample_datasets/sample.pred"
        answer_file = "nagisa/data/sample_datasets/sample.test"

        system_data = nagisa.train.mecab_system_eval.readFile(system_file)
        answer_data = nagisa.train.mecab_system_eval.readFile(answer_file)

        expected_r = [20, 20, 26, 23]
        r = nagisa.train.mecab_system_eval.mecab_eval(system_data, answer_data)

        nagisa.train.mecab_system_eval.print_eval(r)
        self.assertEqual(r, expected_r)


def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(TestNagisa))
    return suite


if __name__ == '__main__':
    unittest.main()
