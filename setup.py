#!/usr/bin/env python3
# encoding: utf-8
"""
setup.py
~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2017-2021 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
from setuptools import setup, find_packages
import re
import os

here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join('leuvenmapmatching', '__init__.py'), 'r') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)
if not version:
    raise RuntimeError('Cannot find version information')


setup(
    name='leuvenmapmatching',
    version=version,
    packages=find_packages(),
    author='Wannes Meert',
    author_email='wannes.meert@cs.kuleuven.be',
    url='https://dtai.cs.kuleuven.be',
    description='Match a trace of locations to a map',
    python_requires='>=3.6',
    license='Apache 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'
    ],
    keywords='map matching',
)
