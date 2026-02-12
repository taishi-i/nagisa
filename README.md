<p align="center"><img width="50%" src="/nagisa/data/nagisa_logo.png" /></p>

---

[![Python package](https://github.com/taishi-i/nagisa/actions/workflows/python-package.yml/badge.svg)](https://github.com/taishi-i/nagisa/actions/workflows/python-package.yml)
[![Coverage Status](https://coveralls.io/repos/github/taishi-i/nagisa/badge.svg?branch=master)](https://coveralls.io/github/taishi-i/nagisa?branch=master)
[![Documentation Status](https://readthedocs.org/projects/nagisa/badge/?version=latest)](https://nagisa.readthedocs.io/en/latest/?badge=latest)
![GitHub License](https://img.shields.io/github/license/taishi-i/nagisa)
[![PyPI](https://img.shields.io/pypi/v/nagisa.svg)](https://pypi.python.org/pypi/nagisa)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/taishi-i/nagisa-demo)
[![Downloads](https://static.pepy.tech/badge/nagisa)](https://pepy.tech/project/nagisa)


Nagisa is a python module for Japanese word segmentation/POS-tagging.

It is designed to be a simple and easy-to-use tool.

This tool has the following features.
-  Based on recurrent neural networks.
-  The word segmentation model uses character- and word-level features [[池田+]](http://www.anlp.jp/proceedings/annual_meeting/2017/pdf_dir/B6-2.pdf).
-  The POS-tagging model uses tag dictionary information [[Inoue+]](http://www.aclweb.org/anthology/K17-1042).

For more details refer to the following links.
-  The documentation is available [here](https://nagisa.readthedocs.io/en/latest/?badge=latest).
-  The article in Japanese is available [here](https://qiita.com/taishi-i/items/5b9275a606b392f7f58e).
-  The presentation slide at PyCon JP (2022) is available [here](https://speakerdeck.com/taishii/pycon-jp-2022).


Installation
=============

You can install nagisa using pip:

```bash
pip install nagisa
````

Supported Platforms:
- 🐧 Linux: Python 3.6 - 3.14
- 🍎 macOS: Python 3.9 - 3.14
- 🪟 Windows: Python 3.9 - 3.14

Basic usage
=============

Sample of word segmentation and POS-tagging for Japanese. The output tokens are normalized by unicodedata.normalize('NFKC', text).

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
import nagisa

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
import nagisa

# default
text = "3月に見た「3月のライオン」"
print(nagisa.tagging(text))
#=> 3/名詞 月/名詞 に/助詞 見/動詞 た/助動詞 「/補助記号 3/名詞 月/名詞 の/助詞 ライオン/名詞 」/補助記号

# If a word ("3月のライオン") is included in the single_word_list, it is recognized as a single word.
new_tagger = nagisa.Tagger(single_word_list=['3月のライオン'])
print(new_tagger.tagging(text))
#=> 3/名詞 月/名詞 に/助詞 見/動詞 た/助動詞 「/補助記号 3月のライオン/名詞 」/補助記号
```

Nagisa provides a built-in Japanese stopwords list. You can use `nagisa.stopwords` to easily filter out common Japanese stopwords (such as particles and auxiliary verbs) from tokenized results.

```python
import nagisa

text = "日本語のストップワードを簡単に利用できます。"
tokens = nagisa.tagging(text)
print(tokens.words)
#=> ['日本', '語', 'の', 'ストップ', 'ワード', 'を', '簡単', 'に', '利用', 'でき', 'ます', '。']

# Filter out stopwords from the tokenized result
words = [word for word in tokens.words if word not in nagisa.stopwords]
print(words)
#=> ['日本', '語', 'ストップ', 'ワード', '簡単', '利用', '。']
```

Train a model
======

Nagisa provides a simple train method
for a joint word segmentation and sequence labeling (e.g, POS-tagging, NER) model.

The format of the train/dev/test files is tsv.
Each line is `word`  and `tag` and one line is represented by `word` \t(tab) `tag`.
Note that you put EOS between sentences.
Refer to [sample datasets](/nagisa/data/sample_datasets) and [tutorial (Train a model for Universal Dependencies)](https://nagisa.readthedocs.io/en/latest/tutorial.html).


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
import nagisa

# After finish training, save the three model files (*.vocabs, *.params, *.hp).
nagisa.fit(train_file="sample.train", dev_file="sample.dev", test_file="sample.test", model_name="sample")

# Build the tagger by loading the trained model files.
sample_tagger = nagisa.Tagger(vocabs='sample.vocabs', params='sample.params', hp='sample.hp')

text = "福岡・博多の観光情報"
words = sample_tagger.tagging(text)
print(words)
#> 福岡/PROPN ・/SYM 博多/PROPN の/ADP 観光/NOUN 情報/NOUN
```
