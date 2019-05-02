=============
Tutorial
=============

Train a Japanese word segmentation and POS tagging model for Universal Dependencies
===================================================================================

Download the dataset
---------------------------

Before we get started,
run the command ``$ pip install nagisa`` to install the nagisa library.
After installing it, download the Japanese UD treebank from UD_Japanese-GDS_.

.. _UD_Japanese-GDS: https://github.com/UniversalDependencies/UD_Japanese-GSD

.. code-block:: bash

    mkdir work
    cd work
    pip install nagisa
    git clone https://github.com/UniversalDependencies/UD_Japanese-GSD


Preprocess and train a model
-----------------------------

First, convert the downloaded files to the input format for nagisa.
The format of the train/dev/test files is tsv.
Each line is word and tag and one line is represented by **word \\t tag**.
Note that you put **EOS** between sentences.

Next,

.. literalinclude:: tutorial_train_ud.py
    :caption: tutorial_train_ud.py
    :name: tutorial_train_ud.py
    :language: python
    :emphasize-lines: 51-52
    :linenos:


.. code-block:: python

    [dynet] random seed: 1234
    [dynet] allocating memory: 32MB
    [dynet] memory allocation done.
    [nagisa] LAYERS: 1
    [nagisa] THRESHOLD: 3
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
    [nagisa] NUM_TRAIN: 7133
    [nagisa] NUM_TEST: 551
    [nagisa] NUM_DEV: 511
    [nagisa] VOCAB_SIZE_UNI: 2352
    [nagisa] VOCAB_SIZE_BI: 25108
    [nagisa] VOCAB_SIZE_WORD: 9143
    [nagisa] VOCAB_SIZE_POSTAG: 17
    Epoch	LR   	Loss 	Time_m	DevWS_f1	DevPOS_f1	TestWS_f1	TestPOS_f1
    1    	0.100	13.37	1.462	91.84   	87.75   	91.63   	87.35
    2    	0.100	6.280	1.473	92.57   	89.67   	92.44   	89.15
    3    	0.100	4.961	1.535	93.54   	90.98   	93.62   	90.18
    4    	0.050	4.256	1.430	92.52   	90.19   	93.62   	90.18
    5    	0.025	3.200	1.443	93.46   	91.06   	93.62   	90.18
    6    	0.025	2.581	1.512	93.56   	91.49   	93.88   	91.29
    7    	0.025	2.379	1.475	93.58   	91.50   	93.73   	91.15
    8    	0.025	2.218	1.476	93.63   	91.57   	93.92   	91.31
    9    	0.025	2.122	1.475	93.78   	91.63   	94.09   	91.40
    10   	0.012	1.985	1.434	93.55   	91.39   	94.09   	91.40



Predict
-------

After finish training,
save the three model files (ja_gsd_ud.vocabs, ja_gsd_ud.params, ja_gsd_ud.hp) in the current directory.

.. literalinclude:: tutorial_predict_ud.py
    :caption: tutorial_predict_ud.py
    :name: tutorial_predict_ud.py
    :language: python
    :linenos:


Error analysis
--------------


.. literalinclude:: tutorial_error_analysis_ud.py


.. code-block:: python

                   AUX  VERB  NOUN  ADV  PRON  PART  PUNCT  SYM  ADJ  PROPN  CCONJ  SCONJ  ADP  NUM  INTJ
           AUX      0    16     2    0     0     0      0    0    2      0      0      1   25    0     0
           VERB    14     0    23    0     1     0      0    0    2      0      0      1    0    0     0
           NOUN     0    12     0    5     1     0      1    0   16    101      0      1    1    2     0
           ADV      0     2     8    0     0     1      0    0    2      1      2      0    0    0     0
           PRON     0     3     6    1     0     0      0    0    1      0      0      0    0    0     0
           PART     1     0     4    0     0     0      0    0    0      0      0      0    0    0     0
           PUNCT    0     0     2    0     0     0      0    2    0      0      0      0    0    0     0
           SYM      0     0     0    0     0     0      0    0    0      0      0      0    0    1     0
           ADJ      8     6    41    3     0     1      0    0    0      4      0      0    0    0     0
           PROPN    0     2    65    0     0     0      0    0    0      0      1      0    0    1     0
           CCONJ    0     0     1    2     0     0      0    0    0      0      0      0    0    0     0
           SCONJ    1     0     1    0     0     0      0    0    0      0      0      0    2    0     0
           ADP      4     0     0    0     0     0      0    0    0      0      0      7    0    0     0
           NUM      0     0     1    0     0     0      0    0    0      0      0      0    0    0     0
           INTJ     0     0     0    1     0     0      0    0    0      0      0      0    0    0     0
