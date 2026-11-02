# -*- coding:utf-8 -*-

"""Smoke test run by cibuildwheel against each built wheel.

Run as a script (not from the project root) so that the installed
package is imported instead of the source tree.
"""

import nagisa

tokens = nagisa.tagging(u'Pythonで簡単に使えるツールです')
assert tokens.words == [u'Python', u'で', u'簡単', u'に', u'使える', u'ツール', u'です'], tokens.words
assert tokens.postags == [u'名詞', u'助詞', u'形状詞', u'助動詞', u'動詞', u'名詞', u'助動詞'], tokens.postags
