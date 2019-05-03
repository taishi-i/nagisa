=============
Basic usage
=============

Word identification is a fundamental step in processing of languages
that have no word boundaries such as Japanese and Chinese.
Nagisa provides a set of functions for Japanese natural language processing.
You can easyily use Japanese word segmentation and POS-tagging
by referring to the sample code below.


Sample code
------------

You can install nagisa with pip.

::

    $ pip install nagisa


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


The Output words can be controlled by POS-tags.

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


Add the user dictionary in easy way.

.. code-block:: python

    # default
    text = "3月に見た「3月のライオン」"
    print(nagisa.tagging(text))
    #=> 3/名詞 月/名詞 に/助詞 見/動詞 た/助動詞 「/補助記号 3/名詞 月/名詞 の/助詞 ライオン/名詞 」/補助記号

    # If a word ("3月のライオン") is included in the single_word_list, it is recognized as a single word.
    new_tagger = nagisa.Tagger(single_word_list=['3月のライオン'])
    print(new_tagger.tagging(text))
    #=> 3/名詞 月/名詞 に/助詞 見/動詞 た/助動詞 「/補助記号 3月のライオン/名詞 」/補助記号



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
