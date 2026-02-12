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


Get stopwords
--------------

Nagisa provides a built-in Japanese stopwords list. You can use `nagisa.stopwords` to easily filter out common Japanese stopwords (such as particles and auxiliary verbs) from tokenized results.


.. code-block:: bash

    python tutorial_stopwords.py


.. literalinclude:: examples/tutorial_stopwords.py
    :caption: tutorial_stopwords.py
    :name: tutorial_stopwords.py
    :language: python
    :linenos:
