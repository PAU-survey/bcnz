from pathlib import Path
import xarray as xr


def cache_model(cache_dir, runs=None):
    """Load models if already run, otherwise run one.
       Args:
           cache_dir (str): Directory storing the models.
           runs (df): Which runs to use. See the config directory.
    """

    import bcnz
    if runs is None:
        runs = bcnz.config.eriksen2019()

    # Ensure all models are run.
    cache_dir = Path(cache_dir)
    for i, row in runs.iterrows():
        path = cache_dir / f'model_{i}.nc'

        if path.exists():
            continue

        print(f'Running for model: {i}')
        model = bcnz.model.model_single(**row)
        model.to_netcdf(path)

    print('Loading the models.')
    D = {}
    for i, row in runs.iterrows():
        path = cache_dir / f'model_{i}.nc'

        # The line of creating a new array is very important. Without
        # this the calibration algorithm became 4.5 times slower.
        f_mod = xr.open_dataset(path).flux
        f_mod = xr.DataArray(f_mod)

        # Store with these entries, but suppress them since
        # they affect calculations.
        f_mod = f_mod.squeeze('EBV')
        f_mod = f_mod.squeeze('ext_law')
        f_mod = f_mod.transpose('z', 'band', 'sed')

        D[i] = f_mod

    return D
