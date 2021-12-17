#!/usr/bin/env python
import datetime

from setuptools import setup, find_packages

VERSION = '0.1'
setup(
    # Metadata
    name='torchmb',
    version=f'{VERSION}_{datetime.datetime.now():%Y%m%d}',
    author='Xitong Gao',
    url='https://github.com/admk/torchmb',
    description='PyTorch Model Batcher.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    packages=find_packages(exclude=('*test*',)),
    zip_safe=True,
    install_requires=['torch', 'einops'],
    classifiers=['Programming Language :: Python :: 3'],
)
