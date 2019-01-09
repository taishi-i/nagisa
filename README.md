<div align="center"><img alt="nagisa" src="/nagisa/data/nagisa_logo.png" width="75%" height="75%"></div>

---

[![Build Status](https://travis-ci.org/taishi-i/nagisa.svg?branch=master)](https://travis-ci.org/taishi-i/nagisa)
[![Documentation Status](https://readthedocs.org/projects/nagisa/badge/?version=latest)](https://nagisa.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/nagisa.svg)](https://pypi.python.org/pypi/nagisa)

Nagisa is a python module for Japanese word segmentation/POS-tagging.
It is designed to be a simple and easy-to-use tool.

This tool has the following features.
- Based on recurrent neural networks.
- The word segmentation model uses character- and word-level features [[池田+]](http://www.anlp.jp/proceedings/annual_meeting/2017/pdf_dir/B6-2.pdf).
- The POS-tagging model uses tag dictionary information [[Inoue+]](http://www.aclweb.org/anthology/K17-1042).

For more details refer to the following links.
- The article in Japanese is available [here](https://qiita.com/taishi-i/items/5b9275a606b392f7f58e).
- The documentation is available [here](https://nagisa.readthedocs.io/en/latest/?badge=latest).

Installation
=============

Python 2.7.x or 3.5+ is required.
This tool uses [DyNet](https://github.com/clab/dynet) (the Dynamic Neural Network Toolkit) to calcucate neural networks.
You can install nagisa by using the following command.
```bash
pip install nagisa
```
If you use nagisa on Windows, please run it with python 3.5+.

Basic usage
======
Sample of word segmentation and POS-tagging for Japanese.

```python
import nagisa

text = 'Pythonで簡単に使えるツールです'
words = nagisa.tagging(text)
print(words)
#=> Python/名詞 で/助詞 簡単/形状詞 に/助動詞 使える/動詞 ツール/名詞 です/助動詞

# Get a list of words
print(words.words)
#=> ['Python', 'で', '簡単', 'に', '使える', 'ツール', 'です']

# Get a list of POS-tags
print(words.postags)
#=> ['名詞', '助詞', '形状詞', '助動詞', '動詞', '名詞', '助動詞']
```

Post-processing functions
=====
Filter and extarct words by the specific POS tags.
```python
# Filter the words of the specific POS tags.
words = nagisa.filter(text, filter_postags=['助詞', '助動詞'])
print(words)
#=> Python/名詞 簡単/形状詞 使える/動詞 ツール/名詞

# Extarct only nouns.
words = nagisa.extract(text, extract_postags=['名詞'])
print(words)
#=> Python/名詞 ツール/名詞

# This is a list of available POS-tags in nagisa.
print(nagisa.tagger.postags)
#=> ['補助記号', '名詞', ... , 'URL']
```

Add the user dictionary in easy way.
```python
# If a word is included in the single_word_list, it is recognized as a single word.
text = "ニューラルネットワークを使ってます。"
tagger_nn = nagisa.Tagger(single_word_list=['ニューラルネットワーク'])
print(tagger_nn.tagging(text))
#=> ニューラルネットワーク/名詞 を/助詞 使っ/動詞 て/助動詞 ます/助動詞 。/補助記号
```


Train a model
======
Nagisa (v0.2.0+) provides a simple train method
for a joint word segmentation and sequence labeling (e.g, POS-tagging, NER) model.


The format of the train/dev/test files is tsv.
Each line is `word`  and `tag` and one line is represented by `word` \t(tab) `tag`.
Note that you put EOS between sentences.


```
$ cat sample.train
唯一	NOUN
の	ADP
趣味	NOU
は	ADP
料理	NOUN
EOS
とても	ADV
おいしかっ	ADJ
た	AUX
です	AUX
。	PUNCT
EOS
ドル	NOUN
は	ADP
主要	ADJ
通貨	NOUN
EOS
```

```python
# After finish training, save the three model files (*.vocabs, *.params, *.hp).
nagisa.fit(train_file="sample.train", dev_file="sample.dev", test_file="sample.test", model_name="sample")

# Build the tagger by loading the trained model files.
sample_tagger = nagisa.Tagger(vocabs='sample.vocabs', params='sample.params', hp='sample.hp')

text = "福岡・博多の観光情報"
words = sample_tagger.tagging(text)
print(words)
#> 福岡/PROPN ・/SYM 博多/PROPN の/ADP 観光/NOUN 情報/NOUN
```

