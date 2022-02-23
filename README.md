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

./bcnz_paudm.py /cephfs/pic.es/astro/scratch/eriksen/output/bcnz/v5_941 /cephfs/pic.es/astro/scratch/eriksen/cache/bcnz/2 941 w3 --ip_dask=tcp://193.109.175.131:45560

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

## Example on mock catalogues
To run on mock catalogues, one needs to provide two catalgoues, one containing the narrowband fluxes and its uncertainties and another with 
the broadband photometry and uncertainties. 
These must be fluxes in PAUS units, which are obtained from from AB magnitudes as mAB = 26 - 2.5*np.log10(flux)

To run on sims you need to run the following command line: 
./bcnz_externalfile.py path_to_nbfiles path_bbfile path_output path_models --ip_dask=dask_ip

The narrowband photometry file needs to be structured as in the PAUS database, with columns: ref_id,band,flux,flux_error	
The narrowband photometry file needs to have the bands and its uncertainty named as (e.g. in the case of COSMOS) ref_id,U,G,R,I,ZN,DU,DG,DR,DI,DZN,
where e.g. DU is the uncertainty in the U-band.
