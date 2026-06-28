# -*- coding:utf-8 -*-

import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        "nagisa_utils",
        ["nagisa/nagisa_utils.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(ext_modules=cythonize(extensions))
