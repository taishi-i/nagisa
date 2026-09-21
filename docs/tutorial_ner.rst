=============================================
Tutorial (Japanese Named Entity Recognition)
=============================================

Train a Japanese NER model for KWDLC
=====================================

This tutorial provides an example of training a Japanese NER model
by using Kyoto University Web Document Leads Corpus(KWDLC).


Download the dataset
--------------------

KWDLC is distributed on GitHub (https://github.com/ku-nlp/KWDLC).
Clone the repository to the working directory.
This tutorial uses version 1.1.1 of KWDLC.

.. code-block:: bash

    git clone --depth 1 --branch v1.1.1 https://github.com/ku-nlp/KWDLC


Install python libraries
-------------------------
Before we get started, please run the following command
to install the libraries used in this tutorial.

.. code-block:: bash

    pip install nagisa
    pip install seqeval


Preprocess the dataset
-----------------------

First, convert the downloaded data to the input data format for nagisa.
The input data format of the train/dev/test files is the tsv format.
The Each line is word and tag and one line is represented by **word \\t tag**.
Note that you put **EOS** between sentences.

In the KNP format files of KWDLC (``KWDLC/knp/``), a named entity is annotated
with a tag such as ``<NE:ORGANIZATION:京都大学>`` on a basic phrase line (a line starting with ``+``).
The code below converts these annotations into word-level tags in the IOB2 format (``B-TYPE``, ``I-TYPE`` and ``O``).
It also splits the corpus into the train/dev/test sets by using the official split of the document IDs (``KWDLC/id/split_for_pas/``).
After running the code, **kwdlc.train**, **kwdlc.dev** and **kwdlc.test** are output to the working directory.


.. code-block:: bash

    python tutorial_preprocess_kwdlc.py


.. literalinclude:: examples/tutorial_preprocess_kwdlc.py
    :caption: tutorial_preprocess_kwdlc.py
    :name: tutorial_preprocess_kwdlc.py
    :language: python
    :linenos:

This is an example of the converted data.

.. code-block:: text

    フランコ	B-PERSON
    ・	I-PERSON
    モスキーノ	I-PERSON
    は	O
    １９５０	B-DATE
    年	I-DATE
    イタリア	B-LOCATION
    に	O
    生まれる	O
    。	O
    EOS


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
    [nagisa] TRAINSET: kwdlc.train
    [nagisa] TESTSET: kwdlc.test
    [nagisa] DEVSET: kwdlc.dev
    [nagisa] DICTIONARY: None
    [nagisa] EMBEDDING: None
    [nagisa] HYPERPARAMS: kwdlc_ner_model.hp
    [nagisa] MODEL: kwdlc_ner_model.params
    [nagisa] VOCAB: kwdlc_ner_model.vocabs
    [nagisa] EPOCH_MODEL: kwdlc_ner_model_epoch.params
    [nagisa] NUM_TRAIN: 12271
    [nagisa] NUM_TEST: 2195
    [nagisa] NUM_DEV: 1585
    [nagisa] VOCAB_SIZE_UNI: 2295
    [nagisa] VOCAB_SIZE_BI: 26344
    [nagisa] VOCAB_SIZE_WORD: 9748
    [nagisa] VOCAB_SIZE_POSTAG: 20
    Epoch	LR   	Loss 	Time_m	DevWS_f1	DevPOS_f1	TestWS_f1	TestPOS_f1
    1    	0.100	6.536	1.596	93.42   	88.54   	95.56   	92.06
    2    	0.100	3.711	1.610	94.39   	89.64   	96.08   	92.70
    3    	0.100	2.912	1.615	94.62   	90.26   	96.46   	93.52
    4    	0.100	2.431	1.676	94.79   	90.80   	96.47   	93.66
    5    	0.100	2.081	1.692	94.83   	90.77   	96.52   	93.82
    6    	0.100	1.768	1.672	95.10   	91.14   	96.51   	93.68
    7    	0.100	1.589	1.892	95.18   	91.31   	96.58   	93.90
    8    	0.050	1.399	1.758	95.00   	90.95   	96.58   	93.90
    9    	0.050	1.014	1.869	95.45   	91.41   	96.70   	93.98
    10   	0.025	0.865	1.591	95.26   	91.22   	96.70   	93.98


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

By checking tag-level accuracy/entity-level micro-f1/classification_report, you can see what the model is wrong with.

.. code-block:: bash

    python tutorial_error_analysis_kwdlc.py


.. literalinclude:: examples/tutorial_error_analysis_kwdlc.py
    :caption: tutorial_error_analysis_kwdlc.py
    :name: tutorial_error_analysis_kwdlc.py
    :language: python
    :linenos:

This is the result of the error analysis on the test set.

.. code-block:: python

    accuracy: 0.9671861495999331
    micro-f1: 0.5798764342453663
                  precision    recall  f1-score   support

        ARTIFACT       0.24      0.27      0.26       114
            DATE       0.79      0.79      0.79       245
        LOCATION       0.69      0.69      0.69       399
           MONEY       0.90      0.90      0.90        20
        OPTIONAL       0.25      0.09      0.13        35
    ORGANIZATION       0.33      0.40      0.36       168
         PERCENT       0.75      0.75      0.75         8
          PERSON       0.50      0.53      0.51       118
            TIME       0.25      0.12      0.17        16

       micro avg       0.57      0.59      0.58      1123
       macro avg       0.52      0.50      0.51      1123
    weighted avg       0.58      0.59      0.58      1123


