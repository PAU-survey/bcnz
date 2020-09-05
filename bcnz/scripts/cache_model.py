#!/usr/bin/env python
# encoding: UTF8

# Script for caching a single model. Useful if running
# the models in parallel.

import fire
from pathlib import Path
import xarray as xr
from IPython.core import debugger as ipdb

def cache_model(cache_dir, i):
    """Load models if already run, otherwise run one.
       Args:
           cache_dir (str): Path where to store the model.
           i (int): Which of the models to calculate.
    """

    import bcnz

    # Hardcoded to the first photo-z paper by now.
    runs = bcnz.config.eriksen2019()


    # Ensure all models are run.
    cache_dir = Path(cache_dir)
    assert cache_dir.exists()

    path = cache_dir / f'model_{i}.nc'

    if path.exists():
        return

    row = runs.iloc[i]
    model = bcnz.model.model_single(**row)
    model.to_netcdf(path)

if __name__ == '__main__':
    fire.Fire(cache_model)
