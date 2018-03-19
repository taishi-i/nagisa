# -*- coding:utf-8 -*-

import os
import sys
import numpy

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

sys.path.append('test')

with open('README.md') as f:
    long_description = f.read()

classifiers = [
    'License :: OSI Approved :: MIT License',
    'Natural Language :: Japanese',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Operating System :: Unix',
    'Topic :: Text Processing :: Linguistic',
    'Topic :: Software Development :: Libraries :: Python Modules'
]

extensions = [Extension('utils', ['nagisa/utils.pyx'])]

setup(
    name = 'nagisa',
    packages=['nagisa'],
    author = 'Taishi Ikeda',
    author_email = 'taishi.ikeda.0323@gmail.com',
    version = '0.0.6',
    description = 'Japanese word segmentation/POS tagging tool based on neural networks',
    long_description = long_description,
    url = 'https://github.com/taishi-i/nagisa',
    download_url = 'https://github.com/taishi-i/nagisa/archive/0.0.6.tar.gz',
    license = 'MIT License',
    platforms = 'Unix',
    install_requires = ['DyNet'],
    classifiers = classifiers,
    include_package_data = True,
    test_suite = 'nagisa_test.suite',
    cmdclass = {"build_ext" : build_ext}, 
    ext_modules = cythonize(extensions),
    include_dirs = [numpy.get_include()],
)
