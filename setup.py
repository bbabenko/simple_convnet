#!/usr/bin/env python

from setuptools import setup

setup(name='SimpleConvnet',
      version='1.0',
      description='A basic implementation of convolutional neural nets',
      author='Boris',
      author_email='bbabenko@gmail.com',
      packages=['simple_convnet'],
      install_requires=[
              'matplotlib',
              'numpy',
              'scipy',
              'scikit-image',
              'scikit-learn',
          ],
     )