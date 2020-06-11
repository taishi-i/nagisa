# -*- coding:utf-8 -*-

import io
import os
import sys

from setuptools import setup
from setuptools.extension import Extension

readme = 'README.md'

try:
    from pypandoc import convert_file
    long_description = convert_file(readme, 'rst')
except ImportError:
    if os.name == 'nt':
        if sys.version_info.major == 2:
            f = io.open(readme, 'r', encoding='utf_8_sig')
        else:
            f = open(readme, 'r', encoding='utf_8_sig')
    else:
        f = open(readme, 'r')
    long_description = f.read()


classifiers = [
    'License :: OSI Approved :: MIT License',
    'Natural Language :: Japanese',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Operating System :: Unix',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    'Topic :: Text Processing :: Linguistic',
    'Topic :: Software Development :: Libraries :: Python Modules'
]


class defer_cythonize(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback

    def c_list(self):
        if self._list is None:
            self._list = self.callback()
        return self._list

    def __iter__(self):
        for elem in self.c_list():
            yield elem

    def __getitem__(self, ii):
        return self.c_list()[ii]

    def __len__(self):
        return len(self.c_list())

def extensions():
    from Cython.Build import cythonize
    import numpy
    extensions = [Extension('utils',
                  ['nagisa/utils.pyx'],
                  include_dirs = [numpy.get_include()])]
    return cythonize(extensions)

setup(
    name = 'nagisa',
    packages=['nagisa'],
    author = 'Taishi Ikeda',
    author_email = 'taishi.ikeda.0323@gmail.com',
    version = '0.2.6',
    description = 'A Japanese tokenizer based on recurrent neural networks',
    long_description = long_description,
    url = 'https://github.com/taishi-i/nagisa',
    download_url = 'https://github.com/taishi-i/nagisa/archive/0.2.6.tar.gz',
    license = 'MIT License',
    platforms = 'Unix',
    setup_requires=['six', 'cython', 'numpy',],
    install_requires = ['six', 'numpy','DyNet'],
    classifiers = classifiers,
    include_package_data = True,
    test_suite = 'test.nagisa_test.suite',
    ext_modules = defer_cythonize(extensions)
)
