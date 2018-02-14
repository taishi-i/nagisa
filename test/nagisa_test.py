# -*- coding:utf-8 -*-

import unittest

import nagisa
tagger = nagisa.Tagger()

class TestNagisa(unittest.TestCase):
    def test_tagging(self):
        text = 'Pythonで簡単に使えるツールです'
        output = 'Python/名詞 で/助詞 簡単/形状詞 に/助動詞 使える/動詞 ツール/名詞 です/助動詞'

        words = tagger.tagging(text)
        self.assertEqual(output, str(words))

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(TestNagisa))
    return suite

if __name__ == '__main__':
    unittest.main()
