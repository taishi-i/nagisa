========================
Japanese Topic Modeling
========================

Run topic modeling using the Livedoor News Corpus
===================================================

This tutorial provides an example of running topic modeling on Japanese news articles
by using `nagisa <https://github.com/taishi-i/nagisa>`_ for tokenization
and scikit-learn's LDA (Latent Dirichlet Allocation).


Install python libraries
-------------------------

Before we get started, please run the following commands
to install the libraries used in this tutorial.

.. code-block:: bash

    pip install nagisa
    pip install scikit-learn
    pip install requests
    pip install tqdm


Run the script
--------------

Run the following command to download the corpus and extract topics.
The `Livedoor News Corpus <https://www.rondhuit.com/download.html#ldcc>`_ (~45 MB)
is downloaded automatically on the first run into the ``ldcc/`` directory.
By default, the model is built from the **title** of each article (``TEXT_TYPE = "title"``).
To use the full article body instead, change the value to ``"body"``.

.. code-block:: bash

    python tutorial_topic_model.py


.. literalinclude:: examples/tutorial_topic_model.py
    :caption: tutorial_topic_model.py
    :name: tutorial_topic_model.py
    :language: python
    :linenos:

This is an example of the output.

.. code-block:: text

    100%|████████████████████████| 7367/7367 [00:28<00:00, 261.03it/s]

    Topic 1:
        スマートフォン, Android, アプリ, iPhone, スマホ, 端末, 対応, 無料, 利用, サービス

    Topic 2:
        映画, 公開, 主演, 出演, 作品, 監督, 俳優, 女優, ドラマ, 役

    Topic 3:
        選手, 試合, チーム, 得点, 優勝, サッカー, 野球, 監督, シーズン, 日本

    Topic 4:
        家電, 発売, 価格, 製品, メーカー, 対応, 機能, 搭載, カメラ, テレビ

    Topic 5:
        女性, 男性, 生活, 仕事, 結婚, 子供, 美容, ファッション, 料理, おすすめ

    Topic 6:
        サービス, 企業, 事業, 市場, 展開, ビジネス, 提供, 投資, 成長, 戦略

    Topic 7:
        政府, 政治, 経済, 日本, 問題, 社会, 国, 対策, 発表, 制度

    Topic 8:
        料理, レシピ, 食材, 食べ, 味, 食事, 野菜, 作り, 簡単, おいしい

    Topic 9:
        音楽, ライブ, アルバム, 曲, アーティスト, 歌, リリース, コンサート, シングル, ツアー

Each topic corresponds to one of the 9 news categories in the Livedoor News Corpus.
By looking at the top words, you can see that the model successfully captures the themes of each category.

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Topic
     - Likely category
     - Key words
   * - Topic 1
     - smax (smartphones)
     - スマートフォン, Android, アプリ, iPhone
   * - Topic 2
     - movie-enter (movies)
     - 映画, 公開, 主演, 監督
   * - Topic 3
     - sports-watch (sports)
     - 選手, 試合, チーム, サッカー, 野球
   * - Topic 4
     - kaden-channel (home appliances)
     - 家電, 発売, 価格, 製品, カメラ
   * - Topic 5
     - peachy / livedoor-homme (lifestyle)
     - 女性, 美容, ファッション, おすすめ
   * - Topic 6
     - it-life-hack (IT / business)
     - 企業, 市場, ビジネス, サービス
   * - Topic 7
     - topic-news (general news)
     - 政府, 政治, 経済, 社会
   * - Topic 8
     - people (food / culture)
     - 料理, レシピ, 食材, 食事
   * - Topic 9
     - movie-enter / people (entertainment)
     - 音楽, ライブ, アルバム, アーティスト
