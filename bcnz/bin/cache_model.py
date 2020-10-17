#!/usr/bin/env python
# encoding: UTF8

# Copyright (C) 2020 Martin B. Eriksen
# This file is part of BCNz <https://github.com/PAU-survey/bcnz>.
#
# BCNz is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BCNz is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BCNz.  If not, see <http://www.gnu.org/licenses/>.

# Script for caching a single model. Useful if running
# the models in parallel.

import fire
from pathlib import Path
import xarray as xr
from IPython.core import debugger as ipdb

def cache_model(model_dir, i):
    """Load models if already run, otherwise run one.

       Args:
           cache_dir (str): Path where to store the model.
           i (int): Which of the models to calculate.
    """

    import bcnz

    # Hardcoded to the first photo-z paper by now.
    runs = bcnz.config.eriksen2019()

    # Ensure all models are run.
    model_dir = Path(model_dir)
    assert model_dir.exists()

    sub = runs.iloc[i:i+1]
    bcnz.model.cache_model(model_dir, sub)
    
if __name__ == '__main__':
    fire.Fire(cache_model)
