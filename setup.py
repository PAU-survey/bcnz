#!/usr/bin/env python
# encoding: UTF8

import glob
import pdb
from setuptools import setup, find_packages

name = 'Martin B. Eriksen'
email = 'martin.b.eriksen@gmail.com'

data_files = [
    ('bcnz/config', glob.glob('bcnz/config/*.yaml')),
    ('bcnz/descr', glob.glob('bcnz/descr/*.yaml'))]

setup(
    name = 'bcnz',
    version = '2',
    packages = find_packages(),

    install_requires = [
        'Numpy >= 1.6',
        'Scipy',
        'Tables',
        'argparse'
    ],
    author = name,
    author_email = email,
    data_files = data_files,
    license = 'Read LICENSE.txt',
    maintainer = name,
    maintainer_email = email,
    #scripts = ['bcnz/bin/bcnz.py'],
)
