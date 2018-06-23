#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension


readme = open('README.rst').read()


requirements = [  ]
dependency_links = [  ]


setup(
    name='mcr',
    version='0.2',
    description='DESCRIPTION',
    long_description=readme + '\n\n' ,
    
    author='Juan Benjumea',
    author_email='jubenjum@gmail.com',
    url='https://github.com/jubenjum/mcr',
    
    packages=['mcr'],
    #package_dir={'': 'src'},

    install_requires=requirements,
    dependency_links=dependency_links,
    
    include_package_data=True,
    data_files=[('config', ['mcr/algorithms.json'])],


    license="GPLv3",
    zip_safe=False,

    scripts = ['bin/segmented_eval.py', 'bin/segmented_predict.py', 
        'bin/segmented_train.py', 'bin/transcription_eval.py',  
        'bin/transcription_predict.py', 'bin/transcription_train.py' ],

    entry_points = {'console_scripts': 
        ['extract_features = mcr.extract_features:main',
        'reduce_features = mcr.reduce_features:main',
        'dump_textgrids = mcr.dump_textgrids:main'], },
 
)
