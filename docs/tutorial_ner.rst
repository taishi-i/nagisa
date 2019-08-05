=============================================
Tutorial (Japanese Named Entity Recognition)
=============================================

Train a Japanese NER model for KWDLC
=====================================

This tutorial provides an example of training a Japanese NER model
by using Kyoto University Web Document Leads Corpus(KWDLC).


Download the dataset
--------------------

Please download KWDLC from http://nlp.ist.i.kyoto-u.ac.jp/EN/index.php?KWDLC manually.
Copy the corpus to the working directory.


Install python libraries
-------------------------
Before we get started, please run the following command
to install the libraries used in this tutorial.

.. code-block:: bash

    pip install nagisa
    pip install seqeval
    pip install beautifulsoup4


Preprocess the dataset
-----------------------

First, convert the downloaded data to the input data format for nagisa.
The input data format of the train/dev/test files is the tsv format.
The Each line is word and tag and one line is represented by **word \\t tag**.
Note that you put **EOS** between sentences.

This preprocess is a little complicated, so please copy the code below and run it.
After running the code, **kwdlc.txt** is output to the working directory.


.. code-block:: bash

    python tutorial_preprocess_kwdlc.py


.. literalinclude:: examples/tutorial_preprocess_kwdlc.py
    :caption: tutorial_preprocess_kwdlc.py
    :name: tutorial_preprocess_kwdlc.py
    :language: python
    :linenos:


Train a model
--------------

Next, you train a NER model by using the ``nagisa.fit()`` function.
After finish training the model, save the three model files (kwdlc_ner_model.vocabs, kwdlc_ner_model.params, kwdlc_ner_model.hp) in the current directory.


.. code-block:: bash

    python tutorial_train_kwdlc.py


.. literalinclude:: examples/tutorial_train_kwdlc.py
    :caption: tutorial_train_kwdlc.py
    :name: tutorial_train_kwdlc.py
    :language: python
    :linenos:

This is a log of the training process.

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
    [nagisa] TRAINSET: kwdlc.train
    [nagisa] TESTSET: kwdlc.test
    [nagisa] DEVSET: kwdlc.dev
    [nagisa] DICTIONARY: None
    [nagisa] EMBEDDING: None
    [nagisa] HYPERPARAMS: kwdlc_ner_model.hp
    [nagisa] MODEL: kwdlc_ner_model.params
    [nagisa] VOCAB: kwdlc_ner_model.vocabs
    [nagisa] EPOCH_MODEL: kwdlc_ner_model_epoch.params
    [nagisa] NUM_TRAIN: 3816
    [nagisa] NUM_TEST: 477
    [nagisa] NUM_DEV: 477
    [nagisa] VOCAB_SIZE_UNI: 1838
    [nagisa] VOCAB_SIZE_BI: 12774
    [nagisa] VOCAB_SIZE_WORD: 4809
    [nagisa] VOCAB_SIZE_POSTAG: 29
    Epoch	LR   	Loss 	Time_m	DevWS_f1	DevPOS_f1	TestWS_f1	TestPOS_f1
    1    	0.100	15.09	0.632	92.41   	83.14   	91.70   	82.63
    2    	0.100	8.818	0.637	93.59   	85.59   	93.21   	85.28
    3    	0.100	6.850	0.637	93.98   	85.60   	93.75   	86.01
    4    	0.100	5.751	0.634	94.44   	87.29   	94.01   	86.99
    5    	0.050	5.028	0.614	94.35   	87.02   	94.01   	86.99
    6    	0.050	3.727	0.647	94.84   	87.52   	94.79   	87.91
    7    	0.025	3.268	0.613	94.52   	87.45   	94.79   	87.91
    8    	0.012	2.761	0.610	94.75   	87.58   	94.79   	87.91
    9    	0.012	2.447	0.634	94.95   	87.79   	95.00   	88.28
    10   	0.006	2.333	0.624	94.73   	87.41   	95.00   	88.28


Predict
-------

You can build the tagger only by loading the three trained model files (kwdlc_ner_model.vocabs, kwdlc_ner_model.params, kwdlc_ner_model.hp) to set arguments in ``nagisa.Tagger()``.

.. code-block:: bash

    python tutorial_predict_kwdlc.py


.. literalinclude:: examples/tutorial_predict_kwdlc.py
    :caption: tutorial_predict_kwdlc.py
    :name: tutorial_predict_kwdlc.py
    :language: python
    :linenos:


Error analysis
--------------

By checking tag-level accuracy/entity-level macro-f1/classification_report, you can see what the model is wrong with.

.. code-block:: bash

    python tutorial_error_analysis_kwdlc.py


.. literalinclude:: examples/tutorial_error_analysis_kwdlc.py
    :caption: tutorial_error_analysis_kwdlc.py
    :name: tutorial_error_analysis_kwdlc.py
    :language: python
    :linenos:

.. code-block:: python

    accuracy: 0.9166868198307134
    macro-f1: 0.5900383141762452
                      precision    recall  f1-score   support

        ARTIFACT       0.33      0.35      0.34        86
        OPTIONAL       0.32      0.19      0.24        31
    ORGANIZATION       0.40      0.33      0.36       109
            DATE       0.84      0.87      0.86       154
        LOCATION       0.64      0.68      0.66       262
           MONEY       0.88      0.88      0.88        16
          PERSON       0.44      0.62      0.51        94
            TIME       0.40      0.44      0.42         9
         PERCENT       0.75      0.50      0.60         6

     avg / total       0.58      0.60      0.59       767


