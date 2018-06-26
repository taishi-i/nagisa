<p align="center"><img width="50%" src="/nagisa/data/nagisa_image.jpg"/></p>

--------------------------------------------------------------------------------

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
- The slide in Japanese is available [here](https://drive.google.com/open?id=1AzR5wh5502u_OI_Jxwsq24t-er_rnJBP).
- The documentation is available [here](https://nagisa.readthedocs.io/en/latest/?badge=latest).

Installation
=============

Python 2.7.x or 3.5+ is required.  
This tool uses [DyNet](https://github.com/clab/dynet) (the Dynamic Neural Network Toolkit) to calcucate neural networks.  
You can install nagisa by using the following command.

```bash
pip install nagisa
```

Usage
======

```python
import nagisa

# Sample of word segmentation and POS-tagging for Japanese
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

# Nagisa gives you a simple word segmentation method.
words = nagisa.wakati(text)
print(words) 
#=> ['Python', 'で', '簡単', 'に', '使える', 'ツール', 'です']
```

