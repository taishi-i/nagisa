# -*- coding:utf-8 -*-

import sys

import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

if sys.platform == "win32":
    extra_compile_args = ["/O2"]
else:
    # The inference kernels in nagisa_utils.pyx rely on the compiler
    # auto-vectorizing their inner loops.
    extra_compile_args = ["-O3"]

extensions = [
    Extension(
        "nagisa_utils",
        ["nagisa/nagisa_utils.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
    )
]

setup(ext_modules=cythonize(extensions))
