
class DummyObject:
    # When scattering a dictionary, Dask ends up doing some weird things..
    def __init__(self):
        self.data = {}

    def __getitem__(self, x):
        return self.data[x]

    def __setitem__(self, x, y):
        self.data[x] = y

    def keys(self):
        return self.data.keys()

def run_photoz_dask(runs, modelD, galcat, output_dir, fit_bands, ip_dask):

    """Run the photo-z on a Dask cluster."""

    path_out = Path(output_dir) / 'pzcat.pq'
    if path_out.exists():
        print('Photo-z catalogue already exists.')
        return

    # If not specified, we start up a local cluster.
    client = Client(ip_dask) if not ip_dask is None else Client()

    # Scatter the model to avoid too much data on the graph. Creating an
    # object is a hack around a weird Dask error.
    tmpD = fix_model(modelD, fit_bands)
    xnew_modelD = DummyObject()
    for key,val in tmpD.items():
        xnew_modelD[key] = val

    xnew_modelD = client.scatter(xnew_modelD)

    galcat = dd.read_parquet(str(output_dir / 'galcat_in.pq'))

    #npartitions = int(302138 / 10) + 1
    npartitions = int(9900 / 10) + 1
    galcat = galcat.reset_index().repartition(npartitions=npartitions).set_index('ref_id')

    ebvD = dict(runs.EBV)


    # To disabled if you want to run a test on a few galaxies without Dask.
#    sub = galcat.head(4)
#    pzcat = bcnz.fit.photoz_flatten(sub, xnew_modelD, ebvD, fit_bands)

#    dask.config.set(scheduler='threads')

    pzcat = galcat.map_partitions(
        bcnz.fit.photoz_flatten, xnew_modelD, ebvD, fit_bands)

#    print('Finished...')

    pzcat = pzcat.repartition(npartitions=100)
    pzcat = dask.optimize(pzcat)[0]

    pzcat.to_parquet(str(path_out))


