# bcnz

The BCNz photo-z code described in [Eriksen 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.4200E/abstract). Support for
the Bayesian extension [Alarcon 2020](https://ui.adsabs.harvard.edu/abs/2020arXiv200711132A/abstract) is pending.


The code performs the following steps:

* Create the model.
* Calibration.
* Run the photo-z.

# Installation
Clone the respository and write

pip install -e .

after entering into the cloned directory.

## Example usage
While intended to be integrated in PAUdm, one can also run the code from the command line. As an 
example, one can run

<<<<<<< HEAD
./bcnz_paudm.py /cephfs/pic.es/astro/scratch/eriksen/output/bcnz/v5_941 /cephfs/pic.es/astro/scratch/eriksen/cache/bcnz/2 941 w3 --ip_dask=tcp://193.109.175.131:45560
=======
./run_bcnz.py /cephfs/pic.es/astro/scratch/eriksen/output/bcnz/v5_941 /cephfs/pic.es/astro/scratch/eriksen/cache/bcnz/7 941 w3 --ip_dask=tcp://193.109.175.131:45560
>>>>>>> 69e8d98e9e99ebe987348a9d1aee6a87d2f0b665

where bcnz_paudm.py is a binary in the bcnz/bin directory. Here the photo-z code will run the code
over MEMBA production 941, which is in the W3 field. Intermediate and final outputs are stored
in the 

/cephfs/pic.es/astro/scratch/eriksen/output/bcnz/v5_941

directory. Further, computing the models are time consuming. If not already existing, they will
be calculated and stored in 

/cephfs/pic.es/astro/scratch/eriksen/cache/bcnz/2.

For running in parallel, the code uses [Dask](https://dask.org/). By specifying Dask, one can
run on an existing Dask cluster. For example, dask-jobqueue supports running Dask on HTCondor,
which is used at the [PIC](www.pic.es) data center.
