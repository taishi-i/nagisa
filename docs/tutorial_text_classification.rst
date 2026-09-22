==========================================
Tutorial (Japanese Text Classification)
==========================================

Classify Japanese news articles with nagisa and scikit-learn
=============================================================

This tutorial provides an example of classifying Japanese news articles into categories
by using nagisa for tokenization and scikit-learn for training a classifier.
You convert the tokenized articles into TF-IDF vectors,
train a logistic regression classifier, evaluate it and classify new texts.


Install python libraries
-------------------------

Before we get started, please run the following commands
to install the libraries used in this tutorial.

.. code-block:: bash

    pip install nagisa
    pip install scikit-learn
    pip install tqdm


Download the dataset
---------------------

This tutorial uses the `Livedoor News Corpus <https://www.rondhuit.com/download.html#ldcc>`_,
which contains 7,367 Japanese news articles in 9 categories.
The corpus (~9 MB) is downloaded automatically into the ``ldcc/`` directory on the first run.
Each document is the title and the body of an article, and its label is the category (the news site of the article).


Tokenize Japanese text with nagisa
-----------------------------------

``tokenize()`` keeps the content words (nouns, verbs, adjectives, etc.) using ``nagisa.extract()``,
lowercases them and removes the stopwords in ``nagisa.stopwords``.
nagisa works on a sentence, so the text is tokenized line by line.

.. literalinclude:: examples/tutorial_text_classification.py
    :pyobject: tokenize
    :language: python

Before splitting the text into words, ``tokenize()`` removes the names of the news sites in ``SOURCE_NAMES``.
Many articles contain the name of their site (e.g., "S-MAX" appears in 863 of the 870 articles of the smax category).
Such words tell the category directly, so a classifier would learn the site names instead of the content of the articles.


Train and evaluate a classifier
--------------------------------

The script splits the corpus into the train set (80%) and the test set (20%),
keeping the ratio of the categories (``stratify=labels``).
The vectorizer and the classifier are fitted on the train set only.
TF-IDF gives a high value to a word that appears frequently in a document but rarely in the other documents.
``TfidfVectorizer`` receives the tokenized documents as they are (``analyzer=identity``)
and ignores the words that appear in only one document (``min_df=2``).
``LogisticRegression`` learns a weight for each word and each category.

Run the following command.
Tokenizing all articles takes several minutes.

.. code-block:: bash

    python tutorial_text_classification.py


.. literalinclude:: examples/tutorial_text_classification.py
    :caption: tutorial_text_classification.py
    :name: tutorial_text_classification.py
    :language: python
    :linenos:

This is the output of tutorial_text_classification.py.
The results may differ with other versions of the libraries (we used nagisa 0.3.0 and scikit-learn 1.9.1).

.. code-block:: text

    train: 5893, test: 1474, features: 34307
    accuracy: 0.8996
                    precision    recall  f1-score   support

    dokujo-tsushin      0.855     0.845     0.850       174
      it-life-hack      0.918     0.897     0.907       174
     kaden-channel      0.948     0.942     0.945       173
    livedoor-homme      0.934     0.696     0.798       102
       movie-enter      0.893     0.960     0.925       174
            peachy      0.785     0.822     0.803       169
              smax      0.923     0.971     0.947       174
      sports-watch      0.927     0.983     0.954       180
        topic-news      0.938     0.890     0.913       154

          accuracy                          0.900      1474
         macro avg      0.902     0.890     0.894      1474
      weighted avg      0.901     0.900     0.899      1474

    dokujo-tsushin  独女, 女性, 女, 人, 結婚, 自分, 3, 男性
    it-life-hack    孫, pc, 見る, 紺子, ロゴ, 製品, クリック, ter
    kaden-channel   話題, 売れ筋, ビデオ, salon, チェック, 発売, パナソニック, hulu
    livedoor-homme  ゴルフ, 転職, 求人, クルマ, ビジネス, 0, 年収, ドバイ
    movie-enter     映画, 公開, 本作, 作品, 劇場, 演じる, 映像, dvd
    peachy          女子, 女性, プレゼント, 恋愛, クリスマス, 事, 肌, 応募
    smax            アプリ, on, android, 執筆, タップ, フォン, 画面, twitter
    sports-watch    選手, 試合, 代表, サッカー, 日本, 野球, 五輪, ダルビッシュ
    topic-news      掲示, ネット, 声, 韓国, 関連, 批判, 記事, 報じ

    新しいAndroidスマートフォンが発売され、アプリの動作も快適になった。
      => smax (0.78)

    日本代表は昨日の試合で2対1で勝利した。
      => sports-watch (0.88)

    監督が新作映画の見どころを語った。
      => movie-enter (0.70)

livedoor-homme has the lowest recall (0.696).
It has the fewest articles (511) and covers many topics such as cars, golf and jobs.
peachy and dokujo-tsushin, which are both about women's lifestyles, also have low F1 scores (0.803 and 0.850).

The words with the largest weights are mostly content words,
such as 映画 and 公開 for movie-enter, and 選手 and 試合 for sports-watch.
Some site-specific words remain: 独女 (a word used by dokujo-tsushin), 紺子 (a column character in it-life-hack),
and 執筆 and on (from "記事執筆：..." and "... smaxjp on Twitter" in the smax articles).

Without removing ``SOURCE_NAMES``, the accuracy is 0.904,
but site names such as peachy, max, sports and watch appear in the top words.
