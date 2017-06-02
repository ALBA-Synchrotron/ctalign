#!/usr/bin/env python

from setuptools import setup, find_packages

# ctalign setup.py to be able to use setuptools

# The version is updated automatically with bumpversion
# Do not update manually
__version = '2.2.2-alpha' 


setup(
    name='ctalign',
    version=__version,
    packages=find_packages(),
    entry_points={
	'console_scripts': [
        'ctalign = alignlib.ctalign:main']
        }
)


