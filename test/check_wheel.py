# -*- coding:utf-8 -*-

"""Smoke test run by cibuildwheel against each built wheel.

Run as a script (not from the project root) so that the installed
package is imported instead of the source tree.
"""

import sys
import sysconfig

import nagisa

if sysconfig.get_config_var("Py_GIL_DISABLED"):
    assert not sys._is_gil_enabled(), "importing nagisa re-enabled the GIL"

tokens = nagisa.tagging(u'Pythonで簡単に使えるツールです')
assert tokens.words == [u'Python', u'で', u'簡単', u'に', u'使える', u'ツール', u'です'], tokens.words
assert tokens.postags == [u'名詞', u'助詞', u'形状詞', u'助動詞', u'動詞', u'名詞', u'助動詞'], tokens.postags
