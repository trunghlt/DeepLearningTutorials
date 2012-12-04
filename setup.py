#!/usr/bin/env python
import sys, os.path
import ConfigParser

try:
    from setuptools import setup
except:
    from distutils.core import setup


setup(name='deep_learning',
      version='0.1',
      description='Lisa Lab Deep Learning Networks',
      long_description=open('README.rst').read(),
      author='Lisa Lab',
      url='https://github.com/trunghlt/DeepLearningTutorials',
      packages=['deep_learning'],
      package_dir={'deep_learning':'code'},
      provides=['deep_learning'],
      license='Unknown',
      install_requires=['theano'])
