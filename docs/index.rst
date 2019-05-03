.. nagisa documentation master file, created by
   sphinx-quickstart on Mon Apr 22 01:27:31 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

nagisa documentation
=====================

Nagisa is a python module for Japanese word segmentation/POS-tagging.
It is designed to be *a simple and easy-to-use tool* for NLP beginners
and has the following features.

- Based on recurrent neural networks.
- The word segmentation model uses character- and word-level features.
- The POS-tagging model uses tag dictionary information.


After installing nagisa by the command ``$ pip install nagisa``,
you can use a Japanese tokenizer and POS tagger in easy way.


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





.. toctree::
   :maxdepth: 2
   :caption: Contents:

   basic_usage
   tutorial
   nagisa_api
