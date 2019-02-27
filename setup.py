from setuptools import setup, find_packages
from glob import glob
from distutils.extension import Extension
from Cython.Distutils import build_ext
from os.path import pathsep
import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = False

# Cython extensions
ext_paths = ('dstrf/*%s', 'dstrf/dsyevh3C/*%s')
if cythonize is False:
    actual_paths = []
    for path in ext_paths:
        actual_paths.extend(glob(path % '.c'))
    ext_modules = [
        Extension(path.replace(pathsep, '.')[:-2], [path])
        for path in actual_paths
    ]
else:
    ext_modules = [Extension(path,
                             [path % '.pyx'],
                             extra_compile_args=['-std=c99'],
                             ) for path in ext_paths]
    ext_modules = cythonize(ext_modules)

setup(
    name="dstrf",
    description="MEG/EEG source localization tool",
    long_description='add-on module to eelbrain for neural TRF estimation '
                     'GitHub: https://github.com/proloyd/DstRF',
    version="0.1",
    python_requires='>=3.6',

    install_requires=[
        'eelbrain',
    ],

    # metadata for upload to PyPI
    author="Proloy DAS",
    author_email="proloy@umd.com",
    license="apache 2.0",
    cmdclass={'build_ext': build_ext},
    include_dirs=[np.get_include()],
    packages=find_packages(),
    ext_modules=ext_modules,
    url='https://github.com/proloyd/DstRF',
    project_urls={
        "Source Code": "https://github.com/proloyd/DstRF",
    }
)
