from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from distutils.core import setup, Extension
from setuptools import setup
import os


descent = Pybind11Extension(
    'optimizer',
    ['lyra/optimizer/optimizer.cpp'],
    include_dirs=['lib/eigen', 'lib/autodiff'],
    extra_compile_args=['-g', '-std=c++11'],
)

descent = Pybind11Extension(
    'glm',
    ['lyra/glm/glm.cpp'],
    include_dirs=['lib/eigen', 'lib/autodiff'],
    extra_compile_args=['-g', '-std=c++11'],
)

cmath = Extension(
    'cmath',
    sources = ['lyra/other/c/cmath.c']
)

setup(
    name = 'MathExtension',
    version='1.0',
    description = 'This is a math package',
    ext_modules=[optimizer, cmath, glm],
    cmdclass={"build_ext": build_ext},
    test_suite='tests'
)

