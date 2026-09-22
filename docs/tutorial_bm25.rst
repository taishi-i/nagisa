==============================================
Tutorial (Japanese BM25 and Document Search)
==============================================

Implement BM25 for Japanese text with nagisa
=============================================

This tutorial provides an example of implementing BM25 for Japanese text
by using nagisa for tokenization and ``nagisa.stopwords`` for removing stopwords.
You extract keywords from news articles with BM25,
and build a simple Japanese document search engine.
Finally, you use nagisa with `BM25S (BM25-Sparse) <https://github.com/xhluca/bm25s>`_,
a fast BM25 library.


Install python libraries
-------------------------

Before we get started, please run the following commands
to install the libraries used in this tutorial.

.. code-block:: bash

    pip install nagisa
    pip install tqdm
    pip install bm25s

Note that bm25s requires Python 3.8 or later.


How document search works
--------------------------

A document search engine returns the documents relevant to a query in order of relevance.
It splits documents and queries into words (terms),
builds an *inverted index* that maps each term to the documents containing it,
computes a score for each document that contains the query terms,
and returns the top-k documents with the highest scores.
Japanese text has no spaces between words, so the first step needs a tokenizer such as nagisa.
BM25 is a standard function for computing this score.


BM25
-----

BM25 gives a high score to a document that contains the query terms many times,
especially the terms that are rare in the corpus.
The score of a document :math:`d` for a query :math:`q` is

.. math::

    \mathrm{score}(q, d) = \sum_{t \in q} \mathrm{IDF}(t) \cdot
    \frac{\mathrm{tf}(t, d)}{\mathrm{tf}(t, d) + k_1 \left(1 - b + b \frac{|d|}{\mathrm{avgdl}}\right)}

.. math::

    \mathrm{IDF}(t) = \ln \left(1 + \frac{N - \mathrm{df}(t) + 0.5}{\mathrm{df}(t) + 0.5}\right)

where :math:`\mathrm{tf}(t, d)` is the frequency of the term :math:`t` in the document :math:`d`,
:math:`\mathrm{df}(t)` is the number of documents containing :math:`t`,
:math:`N` is the number of documents,
:math:`|d|` is the length (the number of terms) of :math:`d`
and :math:`\mathrm{avgdl}` is the average document length.

- :math:`k_1` (usually 1.2-2.0) controls how quickly the score saturates as the term frequency increases.
- :math:`b` (usually 0.75) controls the normalization by the document length.

This is the variant used by Lucene and BM25S.
The original Okapi BM25 also multiplies each summand by :math:`(k_1 + 1)`, which does not change the ranking,
and its IDF can become negative.


Download the dataset
---------------------

This tutorial uses the `Livedoor News Corpus <https://www.rondhuit.com/download.html#ldcc>`_,
which contains 7,367 Japanese news articles in 9 categories.
The corpus (~9 MB) is downloaded automatically into the ``ldcc/`` directory on the first run.
Each document is the title and the body of an article.


Tokenize Japanese text with nagisa
-----------------------------------

``split_words()`` keeps the content words (nouns, verbs, adjectives, etc.) using ``nagisa.extract()``.
nagisa works on a sentence, so the text is tokenized line by line.
The words are lowercased so that a query "iphone" matches "iPhone".
``tokenize()`` then removes the stopwords in ``nagisa.stopwords``.

.. literalinclude:: examples/tutorial_bm25.py
    :pyobject: split_words
    :language: python

.. literalinclude:: examples/tutorial_bm25.py
    :pyobject: tokenize
    :language: python

Particles, auxiliary verbs and symbols are removed based on their POS-tags,
and "し" is removed as a stopword.

.. code-block:: python

    print(nagisa.tagging("iPhoneで使える便利なアプリを紹介します。"))
    #=> iPhone/名詞 で/助詞 使える/動詞 便利/名詞 な/助動詞 アプリ/名詞 を/助詞 紹介/名詞 し/動詞 ます/助動詞 。/補助記号

    print(tokenize("iPhoneで使える便利なアプリを紹介します。"))
    #=> ['iphone', '使える', '便利', 'アプリ', '紹介']


Implement BM25
---------------

The ``BM25`` class builds an inverted index (``term -> {doc_id: term frequency}``)
and the IDF of each term in ``fit()``.
``term_weight()`` computes the BM25 weight of a term in a document, which is the summand of the formula above.

.. literalinclude:: examples/tutorial_bm25.py
    :pyobject: BM25
    :language: python

``keywords()`` ranks the terms of a document by their BM25 weights.
A term gets a high weight when it appears frequently in the document but rarely in the other documents,
so the top terms characterize the document.

``search()`` adds up the weights of the query terms for each document.
Only the documents that contain a query term are scored,
so a query without any known term returns an empty list.


Run the script
---------------

Run the following command to tokenize the corpus, extract keywords and search documents.
Tokenizing all articles takes several minutes.

.. code-block:: bash

    python tutorial_bm25.py


.. literalinclude:: examples/tutorial_bm25.py
    :caption: tutorial_bm25.py
    :name: tutorial_bm25.py
    :language: python
    :linenos:

This is the output of tutorial_bm25.py.

.. code-block:: text

    documents: 7367, vocabulary: 68012

    [1100] スマホでカーナビを実現！純正ドリンクホルダーがスマホホルダーに早変わり【イケショップのレア物】
      ホルダー, holder, ボールジョイント, ドリンク, car, かぶせる, mount, 両側, ミリ, 固定

    [4058] 大人にふさわしいビターな味わい　ショコラケーキで聖夜を飾る
      引き渡し, ビスキュイ, ショコラ, ビター, ヘネシー, ケーキ, ガナッシュ, チョコレート, tel, 濃厚

    [6233] 香川真司、メッシを語る「分かっていても止められない」
      メッシ, シャビ, リオネル, バルセロナ, イニエスタ, マドリード, 切り返し, トリオ, ミラン, レアル

    query: iPhoneの便利なアプリ
      5.55  ようやくiPhoneでもChromeが使えるように！【iPhoneでチャンスを掴め】
      5.31  もっさりiPhoneとはさよなら！ワンタップでサクサクにできる必須アプリ「もっさり解消 for iPhone」【iPhoneアプリ】
      5.09  面倒な画面の明るさ設定はiPhoneを傾けるだけで好みの明るさに！「Tilt de Brightness(明るさ調整)」【iPhoneアプリ】
      5.02  二画面の便利さを450円で！「iDisplay」でiPadをパソコンのセカンドディスプレイ化【iPhoneアプリ】【iPadアプリ】
      5.01  女子だらけの公開イベントに潜入！iPhone女子部がKDDIデザイニングスタジオで「公開リアル部室」を開催【レポート】

    query: サッカー日本代表の試合
      7.56  代表戦皮肉るセルジオ越後氏、「これこそザ・フレンドリーマッチ」
      7.41  U-23日本代表、まさかのドロー。セルジオ氏は「まるで高校サッカーの決勝」
      7.35  日本代表戦を速報する中西哲生さん、新幹線から粘りのツイート
      7.17  日本代表新アウエーユニに「こっちをホームにしろよ」の声
      7.17  【五輪サッカー】 韓国に負けたイギリス代表に英ネットユーザーから批判相次ぐ


Use BM25S (BM25-Sparse) with nagisa
------------------------------------

`BM25S <https://github.com/xhluca/bm25s>`_ is a fast BM25 library
that computes the scores of all documents in advance and stores them in a sparse matrix.
Its default tokenizer splits text at non-word characters such as spaces and punctuation.

.. code-block:: python

    import bm25s

    print(bm25s.tokenize(["日本語の文書を検索します。"], return_ids=False))
    #=> [['日本語の文書を検索します']]

You can use nagisa by passing ``split_words()`` to the ``splitter`` argument of ``bm25s.tokenization.Tokenizer``,
and ``nagisa.stopwords`` to the ``stopwords`` argument.
The following script builds the index with the same tokens as ``tokenize()`` above,
saves the index and the vocabulary of the tokenizer, loads them again and searches documents.
It imports functions from ``tutorial_bm25.py``, so put both scripts in the same directory and run it there.
It also tokenizes the whole corpus, which takes several minutes.

.. code-block:: bash

    python tutorial_bm25s.py


.. literalinclude:: examples/tutorial_bm25s.py
    :caption: tutorial_bm25s.py
    :name: tutorial_bm25s.py
    :language: python
    :linenos:

This is the output of tutorial_bm25s.py.

.. code-block:: text

    [['日本語の文書を検索します']]

    query: iPhoneの便利なアプリ
      5.55  ようやくiPhoneでもChromeが使えるように！【iPhoneでチャンスを掴め】
      5.31  もっさりiPhoneとはさよなら！ワンタップでサクサクにできる必須アプリ「もっさり解消 for iPhone」【iPhoneアプリ】
      5.09  面倒な画面の明るさ設定はiPhoneを傾けるだけで好みの明るさに！「Tilt de Brightness(明るさ調整)」【iPhoneアプリ】
      5.02  二画面の便利さを450円で！「iDisplay」でiPadをパソコンのセカンドディスプレイ化【iPhoneアプリ】【iPadアプリ】
      5.01  女子だらけの公開イベントに潜入！iPhone女子部がKDDIデザイニングスタジオで「公開リアル部室」を開催【レポート】

    query: サッカー日本代表の試合
      7.56  代表戦皮肉るセルジオ越後氏、「これこそザ・フレンドリーマッチ」
      7.41  U-23日本代表、まさかのドロー。セルジオ氏は「まるで高校サッカーの決勝」
      7.35  日本代表戦を速報する中西哲生さん、新幹線から粘りのツイート
      7.17  日本代表新アウエーユニに「こっちをホームにしろよ」の声
      7.17  【五輪サッカー】 韓国に負けたイギリス代表に英ネットユーザーから批判相次ぐ

The results are the same as those of tutorial_bm25.py.
The scores can differ slightly in the last digits because BM25S uses float32.

Notes on using BM25S with nagisa:

- Set ``lower=False``. By default, BM25S lowercases the text before splitting it,
  but nagisa may split and tag lowercased text differently from the original text.
  ``split_words()`` lowercases the words after tokenization instead.
- Pass ``return_as="tuple"`` when tokenizing the corpus so that the index stores the vocabulary.
  Otherwise, queries given as lists of strings get a score of 0 for all documents.
- Save the vocabulary of the tokenizer with ``tokenizer.save_vocab()`` and load it with ``tokenizer.load_vocab()``.
  A new tokenizer has an empty vocabulary, so ``tokenize(queries, update_vocab=False)`` returns no tokens without it.
- Unlike ``search()`` above, BM25S returns ``k`` documents even if a query has no known terms.
  In that case, all the scores are 0, so ignore the results with a score of 0.
