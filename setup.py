#!/usr/bin/env python
# encoding: UTF8

import glob
import pdb
from setuptools import setup, find_packages

# Same author and maintainer.
name = 'Martin B. Eriksen'
email = 'eriksen@pic.es'

setup(
    name = 'bcnz',
    version = '2',
    packages = find_packages(),

    install_requires = [
        'numpy',
        'pandas',
        'tables',
        'xarray',
        'scipy',
        'sklearn',
        'psycopg2',
        'fire',
        'dask',
        'tables',
        'argparse'
    ],
    author = name,
    author_email = email,
    license = 'GPLv3',
    maintainer = name,
    maintainer_email = email,
    scripts = ['bcnz/bin/run_bcnz.py'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
)
