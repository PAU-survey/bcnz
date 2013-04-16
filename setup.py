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
    version = '1',
    packages = find_packages(),

    install_requires = [
        'Python >= 2.7',
        'Numpy',
        'Scipy',
        'Tables'],
    author = name,
    author_email = email,
    data_files = data_files,
    license = 'Read LICENSE.txt',
    maintainer = name,
    maintainer_email = email,
    scripts = ['bcnz/bin/bcnz.py'],
    entry_points = {
        'brownthrower.task': [
          'pzcat = bcnz.tasks.pzcat:pzcat'],
    }
)
