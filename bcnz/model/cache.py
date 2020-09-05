from pathlib import Path
import xarray as xr

def load_cache(cache_dir, runs=None):
    """Load models if already run, otherwise run one."""

    import bcnz
    if runs is None:
        runs = bcnz.config.eriksen2019()

    # Ensure all models are run.
    cache_dir = Path(cache_dir)
    for i,row in runs.iterrows():
        path = cache_dir / f'model_{i}.nc'

        if path.exists():
            continue

        print(f'Running for model: {i}')
        model = bcnz.model.model_single(**row)
        model.to_hdf(path, 'default')

    print('starting to load....')
    D = {}
    for i,row in runs.iterrows():
        path = cache_dir / f'model_{i}.nc'
        
        D[i] = xr.open_dataset(path).flux

    return D
