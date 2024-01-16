================================
Tutorial (Stopwords for nagisa)
================================

How to use stopwords for nagisa
================================

This tutorial provides how to use stopwords for Japanese text in nagisa.


Install python libraries
-------------------------
Before we get started, please run the following command
to install the libraries used in this tutorial.

.. code-block:: bash

    pip install nagisa
    pip install datasets


Get stopwords
--------------

This is a stopword list of frequently used words in the Japanese language,
created according to the tokenization rules of the Japanese text analysis library, nagisa.

This list is constructed by extracting the top 100 most commonly used words
from the CC-100 dataset and Wikipedia.

To access this list of words, simply run the provided program code below.

.. code-block:: bash

    python tutorial_stopwords.py


.. literalinclude:: examples/tutorial_stopwords.py
    :caption: tutorial_stopwords.py
    :name: tutorial_stopwords.py
    :language: python
    :linenos:
