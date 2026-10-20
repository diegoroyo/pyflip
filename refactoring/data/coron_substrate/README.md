# NIRCam coronagraph mask substrate transmission

`jwst_nircam_moda_com_substrate_trans.fits`: intensity transmission of the NIRCam module A coronagraph
mask (COM) substrate, 1.5-5.0 um (columns WAVELENGTH [micron], THROUGHPUT).

- Measured by J. Krist (2015-10-28); the same values are in spaceKLIP as
  `resources/transmissions/ModA_COM_Substrate_Transmission_20151028_JKrist.dat`.
- This file is the STScI ETC throughput table (COMPNAME ModA_COM_Substrate, author B. Brooks,
  2016-09-23), as distributed with webbpsf_ext (`throughputs/`), which uses it in `nircam_com_th`.

STPSF's filter throughputs are imaging-mode curves and do not include it. PyFLIP multiplies it into
the band weights (`dl_utils.apply_substrate_transmission`), when a model directory is generated
(`generate_STPSF_model_params`) and when model files are loaded (`load_model_data`, `load_data`)
unless the directory records it as already applied.
