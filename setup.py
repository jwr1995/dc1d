# -*- coding: utf-8 -*-
# Copyright 2022 Department of Computer Science, The University of Sheffield 
# Author: J. W. Ravenscroft, jwravenscroft1@sheffield.ac.uk

from setuptools import setup, find_packages

setup(
    name = "DeformConv1D",
    version = "1.0",
    packages = find_packages(),

    install_requires = [
        'pytorch',
    ],

    entry_points = {
        'console_scripts': [
            'deformconv1d = nnet.deform_conv:DeformConv1D',
        ]
    }
)

