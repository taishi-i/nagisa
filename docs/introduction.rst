=============
Introduction
=============

Installation
============

You can install nagisa with pip.

::

    pip install nagisa


Usage
=====

A sample of word segmentation and POS-tagging for Japanese.

.. code-block:: python

    import nagisa

    text = 'Pythonで簡単に使えるツールです'
    words = nagisa.tagging(text)
    print(words) 
    #=> Python/名詞 で/助詞 簡単/形状詞 に/助動詞 使える/動詞 ツール/名詞 です/助動詞

    # Get a list of words
    print(words.words) 
    #=> ['Python', 'で', '簡単', 'に', '使える', 'ツール', 'です']

    # Get a list of POS-tags
    print(words.postags) 
    #=> ['名詞', '助詞', '形状詞', '助動詞', '動詞', '名詞', '助動詞']


Output words can be controlled by POS-tags.

.. code-block:: python

    # Extarcting all nouns from a text
    words = nagisa.extract(text, extract_postags=['名詞'])
    print(words) 
    #=> Python/名詞 ツール/名詞

    # Filtering specific POS-tags from a text
    words = nagisa.filter(text, filter_postags=['助詞', '助動詞'])
    print(words) 
    #=> Python/名詞 簡単/形状詞 使える/動詞 ツール/名詞

    # A list of available POS-tags
    print(nagisa.tagger.postags) 
    #=> ['補助記号', '名詞', ... , 'URL']


A word can be recognized as a single word forcibly.

.. code-block:: python

    text = 'ニューラルネットワークを使ってます。'
    print(nagisa.tagging(text)) 
    #=> ニューラル/名詞 ネットワーク/名詞 を/助詞 使っ/動詞 て/助動詞 ます/助動詞 。/補助記号

    # If a word is included in the single_word_list, it is recognized as a single word. 
    tagger_nn = nagisa.Tagger(single_word_list=['ニューラルネットワーク'])
    print(tagger_nn.tagging(text)) 
    #=> ニューラルネットワーク/名詞 を/助詞 使っ/動詞 て/助動詞 ます/助動詞 。/補助記号

Nagisa is good at capturing URLs and emoticons from an input text.

.. code-block:: python

    text = '(人•ᴗ•♡)こんばんは♪'
    words = nagisa.tagging(text)
    print(words) 
    #=> (人•ᴗ•♡)/補助記号 こんばんは/感動詞 ♪/補助記号

    url = 'https://github.com/taishi-i/nagisaでコードを公開中(๑¯ω¯๑)'
    words = nagisa.tagging(url) 
    print(words) 
    #=> https://github.com/taishi-i/nagisa/URL で/助詞 コード/名詞 を/助詞 公開/名詞 中/接尾辞 (๑　̄ω　̄๑)/補助記号

    words = nagisa.filter(url, filter_postags=['URL', '補助記号', '助詞'])
    print(words) 
    #=> コード/名詞 公開/名詞 中/接尾辞
