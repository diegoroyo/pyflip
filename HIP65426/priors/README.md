Data for refactoring/testing_priors.py. Tracked: this file, model_setup/ (the layout written by
model_building.generate_STPSF_model_params, as relative symlinks into ../model_setup) and the pixel_masks.npy symlink.
Not tracked (.gitignore), to be put here before running:
- ote_prior_ckpt_47500.pt, ote_prior_stats.npz: diffusion prior on full OTE OPD maps (InverseBench, EDM-JWST-OPD-OWN) and
  the training-set statistics; used in ote_prior_mode = 'replace'
- ote_drift_prior_ckpt_40000.pt, ote_drift_prior_normalization.npz: diffusion prior on the DRIFT between consecutive WSS maps
  (InverseBench-eddolike-priors export MAST-OTEOPD-drift256hsall; network units x 1.2161e-8 m, 24 c/p low-passed, no mean);
  used in mode 'offset' on top of the WSS map, for the Lyot-plane OPD on top of the static NIRCam OPD, and in any DPS window on
  the science frame
- ote_field_term_NRCA5_MASK335R.npy: stpsf's field-dependent OTE WFE at the coronagraph (InverseBench/scripts/make_ote_field_term.py),
  needed in 'replace' mode only
- ../drive_data/HIP65426/: the original FLIP calints (CAL_VER 1.18.1, identical to the old demo's npy frames) read by the script.
  calints/ here is a MAST reprocessing (CAL_VER 2.0.1, 2026-08) NOT used: per-integration background offset (up to 2 MJy/sr) and NaNs.
- outputs/: where the script writes
