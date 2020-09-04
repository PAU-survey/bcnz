from pathlib import Path

def load_cache(cache_dir, runs=None):
    """Load models if already run, otherwise run one."""

    import bcnz
    if runs is None:
        runs = bcnz.config.eriksen2019()

    cache_dir = Path(cache_dir)
    for i,row in runs.iterrows():
        path = cache_dir / f'model_{i}.h5'

        if path.exists():
            continue

        print(f'Running for model: {i}')
        model = bcnz.model.model_single(**row)
        model.to_hdf(path, 'default')
