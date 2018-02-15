![Alt text](/nagisa/data/nagisa_image.jpg 'An image of title')

# nagisa

Nagisa is a python module for Japanese word segmentation/POS-tagging.  
It is designed to be a simple and easy-to-use tool.  

This tool has the following features.
- Based on recurrent neural networks. 
- The word segmentation model uses character- and word-level features [[池田+]](http://www.anlp.jp/proceedings/annual_meeting/2017/pdf_dir/B6-2.pdf).
- The POS-tagging model uses tag dictionary information [[Inoue+]](http://www.aclweb.org/anthology/K17-1042).

Requirements
========
[DyNet](https://github.com/clab/dynet) (Neural Network Toolkit) is required.  
Nagisa is compatible with: Python 2.7-3.6.

Installation
========

```bash
# From github
git clone https://github.com/taishi-i/nagisa
cd nagisa
# If you got a permission denied error, 
# please run the following line.
# sudo python setup.py install
python setup.py install
```


Usage
====

```python
import nagisa
tagger = nagisa.Tagger()

# Sample of word segmentation and POS-tagging for Japanese
text = 'Pythonで簡単に使えるツールです'
words = tagger.tagging(text)
print(words) # Python/名詞 で/助詞 簡単/形状詞 に/助動詞 使える/動詞 ツール/名詞 です/助動詞

# Get a list of words
print(words.words) # ['Python', 'で', '簡単', 'に', '使える', 'ツール', 'です']
# Get a list of POS-tags
print(words.postags) # ['名詞', '助詞', '形状詞', '助動詞', '動詞', '名詞', '助動詞']

# A list of available POS-tags
print(tagger.postags) # ['補助記号', '名詞', ... , 'URL']

# Extarcting all nouns from a text
words = tagger.extract(text, ['名詞'])
print(words) # Python/名詞 ツール/名詞

# Filtering specific POS-tags from a text
words = tagger.filter(text, ['助詞', '助動詞'])
print(words) # Python/名詞 簡単/形状詞 使える/動詞 ツール/名詞
```


Feature
====

```python
# Nagisa is good at capturing URLs and emoticons from a text.
text = '(人•ᴗ•♡)こんばんは♪'
words = tagger.tagging(text)
print(words) # (人•ᴗ•♡)/補助記号 こんばんは/感動詞 ♪/補助記号

url = 'https://github.com/taishi-i/nagisaでコードを公開中(๑¯ω¯๑)'
words = tagger.tagging(url) 
print(words) # https://github.com/taishi-i/nagisa/URL で/助詞 コード/名詞 を/助詞 公開/名詞 中/接尾辞 (๑　̄ω　̄๑)/補助記号

words = tagger.filter(url, ['URL', '補助記号', '助詞'])
print(words) # コード/名詞 公開/名詞 中/接尾辞
```
