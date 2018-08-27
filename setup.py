#!/usr/bin/env python3
# encoding: utf-8
"""
setup.py
~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2017-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
from setuptools import setup, Command, find_packages
from setuptools.command.sdist import sdist
import re
import os

here = os.path.abspath(os.path.dirname(__file__))


class MySDistCommand(sdist):
    def run(self):
        PrepReadme.run_pandoc()
        super().run()


class PrepReadme(Command):
    description = "Translate readme from Markdown to ReStructuredText"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        PrepReadme.run_pandoc()

    @staticmethod
    def run_pandoc():
        import subprocess as sp
        try:
            sp.call(['pandoc', '--from=markdown', '--to=rst', '--output=README', 'README.md'])
        except sp.CalledProcessError as err:
            print('Running pandoc failed')
            print(err)
        except FileNotFoundError as err:
            print('Running pandoc failed')
            print(err)


with open(os.path.join('leuvenmapmatching', '__init__.py'), 'r') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)
if not version:
    raise RuntimeError('Cannot find version information')


readme_path = os.path.join(here, 'README')
if not os.path.exists(readme_path):
    PrepReadme.run_pandoc()
try:
    with open(readme_path, 'r') as f:
        long_description = f.read()
except FileNotFoundError as err:
    long_description = ""
    print("No Readme found")
    print(err)


setup(
    name='leuvenmapmatching',
    version=version,
    packages=find_packages(),
    author='Wannes Meert',
    author_email='wannes.meert@cs.kuleuven.be',
    url='https://dtai.cs.kuleuven.be',
    description='Match a trace of locations to a map',
    long_description=long_description,
    install_requires=['numpy', 'scipy'],
    extras_require={
        'vis': ['smopy', 'matplotlib>=2.0.0'],
        'db': ['rtree', 'pyproj'],
        'all': ['smopy', 'matplotlib>=2.0.0', 'rtree', 'pyproj', 'nvector==0.5.2', 'gpxpy', 'pykalman']
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    python_requires='>=3.6',
    license='Apache 2.0',
    classifiers=(
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'
    ),
    keywords='map matching',
    cmdclass={
        'readme': PrepReadme,
        'sdist': MySDistCommand,
    },
)
