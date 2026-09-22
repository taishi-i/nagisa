=============
Tutorial
=============

Train a Japanese word segmentation and POS tagging model for Universal Dependencies
===================================================================================

This tutorial provides an example of training a joint word segmentation and POS tagging model by using Japanese universal dependencies treebank.
You get to know how to build the original sequence labeling model through this tutorial.

Download the dataset
--------------------

Before we get started,
please run the command ``$ pip install nagisa`` to install the nagisa library.
After installing it, download the Japanese UD treebank from UD_Japanese-GSD_.

.. _UD_Japanese-GSD: https://github.com/UniversalDependencies/UD_Japanese-GSD

.. code-block:: bash

    mkdir work
    cd work
    pip install nagisa
    git clone https://github.com/UniversalDependencies/UD_Japanese-GSD

Preprocess the dataset and train a model
----------------------------------------

First, convert the downloaded data to the input data format for nagisa.
The input data format of the train/dev/test files is tsv.
The Each line is word and tag and one line is represented by **word \\t tag**.
Note that you put **EOS** between sentences.
Refer to `the tiny sample datasets <https://github.com/taishi-i/nagisa/tree/master/nagisa/data/sample_datasets>`_.

Next, you train a joint word segmentation and POS-tagging model by using the ``nagisa.fit()`` function. After finish training the model, save the three model files (ja_gsd_ud.vocabs, ja_gsd_ud.params, ja_gsd_ud.hp) in the current directory.

.. literalinclude:: examples/tutorial_train_ud.py
    :caption: tutorial_train_ud.py
    :name: tutorial_train_ud.py
    :language: python
    :linenos:

This is a log of the training process.

.. code-block:: python

    [nagisa] LAYERS: 1
    [nagisa] THRESHOLD: 2
    [nagisa] DECAY: 1
    [nagisa] EPOCH: 10
    [nagisa] WINDOW_SIZE: 3
    [nagisa] DIM_UNI: 32
    [nagisa] DIM_BI: 16
    [nagisa] DIM_WORD: 16
    [nagisa] DIM_CTYPE: 8
    [nagisa] DIM_TAGEMB: 16
    [nagisa] DIM_HIDDEN: 100
    [nagisa] LEARNING_RATE: 0.1
    [nagisa] DROPOUT_RATE: 0.3
    [nagisa] SEED: 1234
    [nagisa] TRAINSET: ja_gsd_ud.train
    [nagisa] TESTSET: ja_gsd_ud.test
    [nagisa] DEVSET: ja_gsd_ud.dev
    [nagisa] DICTIONARY: None
    [nagisa] EMBEDDING: None
    [nagisa] HYPERPARAMS: ja_gsd_ud.hp
    [nagisa] MODEL: ja_gsd_ud.params
    [nagisa] VOCAB: ja_gsd_ud.vocabs
    [nagisa] EPOCH_MODEL: ja_gsd_ud_epoch.params
    [nagisa] NUM_TRAIN: 7050
    [nagisa] NUM_TEST: 543
    [nagisa] NUM_DEV: 507
    [nagisa] VOCAB_SIZE_UNI: 2340
    [nagisa] VOCAB_SIZE_BI: 24792
    [nagisa] VOCAB_SIZE_WORD: 8724
    [nagisa] VOCAB_SIZE_POSTAG: 17
    Epoch	LR   	Loss 	Time_m	DevWS_f1	DevPOS_f1	TestWS_f1	TestPOS_f1
    1    	0.100	11.37	0.584	96.57   	92.69   	96.85   	92.32
    2    	0.100	4.656	0.585	97.04   	93.64   	97.63   	94.04
    3    	0.100	3.543	0.589	97.60   	94.64   	97.66   	94.56
    4    	0.050	2.930	0.573	97.37   	94.69   	97.66   	94.56
    5    	0.050	2.075	0.666	97.72   	95.06   	98.08   	95.50
    6    	0.025	1.786	0.572	97.54   	95.11   	98.08   	95.50
    7    	0.025	1.451	0.587	97.79   	95.36   	98.13   	95.62
    8    	0.025	1.325	0.586	97.79   	95.28   	98.14   	95.61
    9    	0.025	1.247	0.589	97.80   	95.44   	98.13   	95.71
    10   	0.012	1.204	0.570	97.76   	95.27   	98.13   	95.71

Predict
-------

You can build the tagger only by loading the three trained model files (ja_gsd_ud.vocabs, ja_gsd_ud.params, ja_gsd_ud.hp) to set arguments in ``nagisa.Tagger()``.

.. literalinclude:: examples/tutorial_predict_ud.py
    :caption: tutorial_predict_ud.py
    :name: tutorial_predict_ud.py
    :language: python
    :linenos:

Error analysis
--------------

By checking a confusion matrix,
you can see what the model is wrong with.
The code shows how to create a confusion matrix by comparing the predicted tags with the gold-standard tags.

.. literalinclude:: examples/tutorial_error_analysis_ud.py
    :caption: tutorial_error_analysis_ud.py
    :name: tutorial_error_analysis_ud
    :language: python
    :linenos:

This is a confusion matrix if tagger make a mistake in prediction.
This confusion matrix shows that the tagger often confuses "NOUN" and "PROPN"
(NOUN is predicted as PROPN 60 times, and PROPN as NOUN 50 times)
in this UD_Japanese-GSD dataset.

.. code-block:: text

           ADJ  ADV  NOUN  PROPN  AUX  PART  DET  ADP  VERB  SCONJ  PRON  PUNCT  CCONJ  NUM  SYM  INTJ
    ADJ      0    2    32      2   11     0    0    1     5      0     0      0      0    0    0     0
    ADV      4    0    23      4    0     0    0    0     4      0     0      0      1    0    0     0
    NOUN    13    5     0     60    2     0    0    1    14      1     0      0      0    0    0     0
    PROPN    0    0    50      0    0     0    0    0     1      0     0      0      0    1    0     0
    AUX      4    0     0      0    0     1    0   19     6      1     0      0      0    0    0     0
    PART     1    0     4      1    2     0    0    4     1      0     0      0      0    0    0     0
    DET      0    0     0      0    0     0    0    2     2      0     0      0      0    0    0     0
    ADP      0    2     2      0   13     4    0    0     0      3     0      0      0    0    0     0
    VERB     2    2    19      0    4     0    1    0     0      2     0      0      0    0    0     0
    SCONJ    0    0     2      0    5     0    0    4     1      0     0      0      0    0    0     0
    PRON     0    0     5      0    0     0    0    0     2      0     0      0      0    0    0     0
    PUNCT    0    0     2      0    0     0    0    0     0      0     0      0      0    0    0     0
    CCONJ    0    0     0      0    2     0    0    0     0      0     0      0      0    0    0     0
    NUM      0    0     3      0    0     0    0    0     0      0     0      0      0    0    0     0
    SYM      0    0     1      0    0     0    0    0     0      0     0      0      0    0    0     0
    INTJ     0    0     0      0    0     0    0    0     1      0     0      0      0    0    0     0
