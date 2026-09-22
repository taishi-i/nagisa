====================================
Tutorial (Japanese Topic Modeling)
====================================

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
The `Livedoor News Corpus <https://www.rondhuit.com/download.html#ldcc>`_ (~9 MB)
is downloaded automatically on the first run into the ``ldcc/`` directory.
By default, the model is built from the **title** of each article (``TEXT_TYPE = "title"``).
To use the article body instead, see `Use the article body`_.

.. code-block:: bash

    python tutorial_topic_model.py


.. literalinclude:: examples/tutorial_topic_model.py
    :caption: tutorial_topic_model.py
    :name: tutorial_topic_model.py
    :language: python
    :linenos:

This is an example of the output.
The results may differ with other versions of the libraries (we used nagisa 0.3.0 and scikit-learn 1.9.1).

.. code-block:: text

    Topic 1:
        女子, 世界, 発言, オトナ, 理由, あなた, 恋愛, 時代, 美人, 独女

    Topic 2:
        発売, 発表, 終了, プレゼント, 登場, 代表, デジ通, ビデオ, xperia, 生活

    Topic 3:
        さん, 写真, 知っ, 虎の巻, すぎる, 結婚, 語る, ススメ, 説教, 自分

    Topic 4:
        映画, アプリ, vol, レビュー, 特集, 独女, 選手, 年収, ゴルフ, 部屋

    Topic 5:
        watch, sports, 日本, 韓国, ネット, vol, 批判, by, 東京, cafe

    Topic 6:
        android, チェック, 対応, ドコモ, スマホ, 売れ筋, 提供, 搭載, ニュース, 向け

    Topic 7:
        話題, ntt, スマート, フォン, テレビ, インタビュー, ドコモ, 問題, 利用, 社長

    Topic 8:
        iphone, 公開, ランキング, 動画, 決定, 女性, 開催, 殺到, 出演, ゲーム

    Topic 9:
        開始, レポート, 監督, モデル, 人気, サービス, google, キャンペーン, akb, 好き

Article titles are short, so many topics mix the words of several categories.
Some topics have a clear subject.
For example, Topic 6 (android, チェック, 対応, ドコモ, スマホ) is about smartphones.
Topic 5 (watch, sports, 日本, 韓国) is about sports, but "watch" and "sports" come from
the tag 【Sports Watch】 at the beginning of many titles in the sports-watch category.


Use the article body
--------------------

Set ``TEXT_TYPE = "body"`` and run the script again.
Tokenizing the article bodies takes several minutes.
This is an example of the output with the article body.

.. code-block:: text

    Topic 1:
        紹介, 記事, 写真, 画面, アプリ, チェック, 使っ, facebook, 表示, 便利

    Topic 2:
        さん, 自分, 女性, いい, 結婚, 仕事, 男性, 思っ, 思い, どう

    Topic 3:
        ネット, 番組, 放送, テレビ, 関連, 記事, 選手, 掲示, 話題, 情報

    Topic 4:
        プレゼント, 料理, ゴルフ, 東京, 女性, アイテム, 商品, クリスマス, 効果, さん

    Topic 5:
        映画, 公開, 作品, 世界, 監督, 本作, 映像, 撮影, 日本, ドラマ

    Topic 6:
        女子, 人気, さん, 彼女, 女性, ファッション, cm, 登場, モデル, ちゃん

    Topic 7:
        日本, 韓国, 代表, 写真, 世界, akb, チーム, 関連, 監督, 話題

    Topic 8:
        スマート, フォン, android, アプリ, 更新, max, 利用, サービス, 機能, ソフトウェア

    Topic 9:
        発売, 搭載, 対応, モデル, カメラ, 発表, 製品, 機能, バッテリー, サイズ

The table shows, for the model built from the article body, how many articles have each topic
as their most probable topic (the argmax of ``lda.transform(dtm)``) and the most common categories among them.
The script does not print these counts.

.. list-table::
   :header-rows: 1
   :widths: 10 10 35 45

   * - Topic
     - Articles
     - Most common categories (share)
     - Top 5 words
   * - Topic 1
     - 342
     - it-life-hack (48%)
     - 紹介, 記事, 写真, 画面, アプリ
   * - Topic 2
     - 1,377
     - dokujo-tsushin (50%)
     - さん, 自分, 女性, いい, 結婚
   * - Topic 3
     - 1,092
     - topic-news (44%), sports-watch (38%)
     - ネット, 番組, 放送, テレビ, 関連
   * - Topic 4
     - 749
     - peachy (58%)
     - プレゼント, 料理, ゴルフ, 東京, 女性
   * - Topic 5
     - 984
     - movie-enter (77%)
     - 映画, 公開, 作品, 世界, 監督
   * - Topic 6
     - 218
     - peachy (31%)
     - 女子, 人気, さん, 彼女, 女性
   * - Topic 7
     - 673
     - sports-watch (49%)
     - 日本, 韓国, 代表, 写真, 世界
   * - Topic 8
     - 1,166
     - smax (53%)
     - スマート, フォン, android, アプリ, 更新
   * - Topic 9
     - 766
     - kaden-channel (37%)
     - 発売, 搭載, 対応, モデル, カメラ

Some topics match a category well, such as movie-enter for Topic 5 (77%) and smax for Topic 8
("max" comes from "S-MAX"). Topics 3, 6 and 9 mix several categories.
