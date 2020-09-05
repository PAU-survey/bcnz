# encoding: UTF8

# Simple wrapper to cache the output.
from pathlib import Path
import pandas as pd

def cache_zp(run_dir, *args, **kwds):
    """Functionality for caching the zero-points."""

    run_dir = Path(run_dir)
    path = run_dir / 'zp.h5'

    import bcnz
    if not path.exists():
        zp = bcnz.calib.calib(*args, **kwds)
        zp.to_hdf(path, 'default')
    else:
        zp = pd.read_hdf(path, 'default')

    return zp