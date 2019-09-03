from setuptools import setup, find_packages
from glob import glob
from distutils.extension import Extension
# from Cython.Distutils import build_ext
from os.path import pathsep
import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = False

# Cython extensions
ext = '.pyx' if cythonize else '.c'
ext_cpp = '.pyx' if cythonize else '.cpp'
extensions = [
    Extension('dstrf.opt', [f'dstrf/opt{ext}']),
    Extension('dstrf.dsyevh3C.dsyevh3py', [f'dstrf/dsyevh3C/dsyevh3py{ext_cpp}'], include_dirs=['dsyevh3C']),
]
if cythonize:
    extensions = cythonize(extensions)


setup(
    name="dstrf",
    description="MEG/EEG source localization tool",
    long_description='add-on module to eelbrain for neural TRF estimation'
                     'GitHub: https://github.com/proloyd/DstRF',
    version="0.3dev",
    python_requires='>=3.6',

    install_requires=[
        'eelbrain',
    ],

    # metadata for upload to PyPI
    author="Proloy DAS",
    author_email="proloy@umd.com",
    license="apache 2.0",
    # cmdclass={'build_ext': build_ext},
    include_dirs=[np.get_include()],
    packages=find_packages(),
    ext_modules=extensions,
    url='https://github.com/proloyd/DstRF',
    project_urls={
        "Source Code": "https://github.com/proloyd/DstRF/archive/0.2.tar.gz",
    }
)
