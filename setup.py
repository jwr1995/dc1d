# -*- coding: utf-8 -*-
# Copyright 2022 Department of Computer Science, The University of Sheffield 
# Author: J. W. Ravenscroft, jwravenscroft1@sheffield.ac.uk

from setuptools import setup, find_packages

setup(
    name = "dc1d",
    version = "1.0",
    # packages = find_packages(),


    entry_points = {
        'console_scripts': [
            'DeformConv1D = dc1d.nnet:DeformConv1D',
            'linterpolate = d1cd.ops:linterpolate'
        ]
    }
)

