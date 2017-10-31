#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension


readme = open('README.rst').read()

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]


setup(
    name='mcr',
    version='0.1',
    description='DESCRIPTION',
    long_description=readme + '\n\n' ,
    author='Juan Benjumea',
    author_email='jubenjum@gmail.com',
    url='https://github.com/jubenjum/mcr',
    packages=[ 'mcr' ],
    package_dir={'':
                 'src'},
    include_package_data=True,
    install_requires=requirements,
    license="GPLv3",
    zip_safe=False,
    keywords='tde',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GPLv3 License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
)
