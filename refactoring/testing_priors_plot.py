# testing_priors.py plus logging of the OTE OPD and of the injected planet over the iterations, for comparing runs
# (figures: priors/plots.py). Usage: python testing_priors_plot.py <variant>, the variant being a key of VARIANTS below
# (overrides of the settings); results go to pyflip-nw/outputs/<variant>/metrics.npz.
import sys
from wavefronts import BroadbandWavefront
from model_building import assemble_JWST_NIRCam_coron
from priors.model import OPDPrior
from priors.dpsadam import DPSAdamOPD, DPSPhase
from priors.stages import prior_stages
from priors.grids import CoarseOPDGrid, FluxScale
from priors.plots import opd_bandpass, opd_rms_nm
from optimization_handlers import default_stages
import torch.nn as nn
import torch 
import os
from dl_utils import load_model_data, load_and_crop_data, inject_fake_planets, hardcoded_match_flux,track_SNR, calc_snr, circular_mask
import numpy as np
import tqdm
import torch.nn.functional as F

### TODO:
#- Track SNR
#- save calibrated model for future reference
#- Visualization tools
# Test on other data

#### PRIOR SETTINGS

USE_PRIOR = True

priors_dir = '/srv/tmp/droyo/exoplanet/pyflip-nw/HIP65426/priors'
calints_dir = '/srv/tmp/droyo/exoplanet/pyflip-nw/HIP65426/drive_data/HIP65426' # the original FLIP calints (pipeline 1.18.1, no NaNs; the MAST 2.0.1 files carry a background offset)
# The two priors (checkpoint, training statistics, network hyperparameters of the checkpoint): on full OTE OPD maps (used in
# mode 'replace') and on the drift between consecutive WSS maps (mode 'offset', and any DPS window on the science frame)
ote_map_prior = dict(ckpt_path=priors_dir + '/ote_prior_ckpt_47500.pt', stats_path=priors_dir + '/ote_prior_stats.npz', kind='map',
                     net_kwargs=dict(img_resolution=256, img_channels=1, model_channels=64, channel_mult=[1, 1, 2, 2, 4], attn_resolutions=[16], num_blocks=1))
ote_drift_prior = dict(ckpt_path=priors_dir + '/ote_drift_prior_ckpt_40000.pt', stats_path=priors_dir + '/ote_prior_stats.npz', kind='drift',
                       normalization_path=priors_dir + '/ote_drift_prior_normalization.npz',
                       net_kwargs=dict(img_resolution=256, img_channels=1, model_channels=64, channel_mult=[1, 1, 2, 2, 4], attn_resolutions=[16], num_blocks=2))

# Phases of the OTE OPD (iterations): the WFS map as in testing.py before ref_ote_prior_start, DPS-Adam between
# ref_ote_prior_start and ref_ote_prior_end (one diffusion step per iteration), frozen + pyFLIP's own optimization of a
# residual grid afterwards. 2000 diffusion steps, as InverseBench's tuned DPS-Adam settings.
ref_ote_prior_start, ref_ote_prior_end = 600, 2600
ote_prior_mode = 'replace' # 'replace': the map prior models the full OTE OPD and the WFS map is ignored; 'offset': the drift prior's output is added to the WFS map

# Optional second DPS window on the science frame with the drift prior (a drift on top of what the reference stage gave);
# the residual grid is frozen during it and refined afterwards (stage 2.2, priors/stages.py). None = off.
sci_ote_prior_start, sci_ote_prior_end = None, None # e.g. 6100, 8100
ote_guidance_scale = 5e-3 # learning rate of the latent in the optimizer (network units; 1 unit = 2 std of the training maps = 47.5 nm)
ote_grid_lr = 4e-9 # learning rate of the residual OTE grid fitted by pyFLIP after ref_ote_prior_end (testing.py: 4e-9, same group as the Lyot OPD)
ote_grid_frozen_until = None # iteration until which the residual grid is frozen (None: ref_ote_prior_end); e.g. 11001 = never fitted
ote_fit_grid_in_science = None # refine the residual grid in stage 2.2 (priors/stages.py); None = only when a science DPS window is set
ote_prior_sigma_max, ote_prior_sigma_min = 10., 0.002 # noise schedule
ote_prior_sde = True # reverse SDE (noise at each step) or probability-flow ODE
ote_prior_seed = 0

# stpsf's fixed field-dependent OTE WFE at the coronagraph field point (30 nm rms; InverseBench/scripts/make_ote_field_term.py).
# The WFS map of the data folder already contains it (stpsf adds it when the OTE plane is exported at the coronagraph
# aperture); the prior's training maps are at the sensing field point and do not. Needed in 'replace' mode only.
ote_field_term_path = priors_dir + '/ote_field_term_NRCA5_MASK335R.npy'

# Lyot-plane (NIRCam) OPD: 'free' (testing.py: full-resolution free grid), 'frozen' (static NIRCam OPD only), 'coarse' (free
# grid of lyot_coarse_nodes x lyot_coarse_nodes nodes upsampled by lyot_coarse_upsample, priors/grids.py: band-limited to half
# that many cycles per array) or 'drift' (the OTE drift prior in mode 'offset' on top of the static NIRCam OPD, DPS window
# lyot_prior_start-lyot_prior_end, same guidance scale and schedule as the OTE; the Lyot free grid is then never fitted).
# lyot_grid_lr: learning rate of the Lyot grid (free or coarse) when not None (None: testing.py's OPD group, 4e-9)
lyot_mode = 'free'
lyot_coarse_nodes = 24
lyot_coarse_upsample = 'bicubic' # round four runs; 'fourier' from round five
lyot_grid_lr = None
lyot_grid_frozen_until = None # iteration before which the Lyot grid (free or coarse) gets no update, e.g. 6100 = only fitted on the science frame
lyot_prior_start, lyot_prior_end = 600, 5500
lyot_sci_prior_start, lyot_sci_prior_end = None, None # optional second DPS window of the Lyot drift prior on the science frame, e.g. 6100, 8100
lyot_guidance_scale, lyot_sci_guidance_scale = None, None # guidance scale of the Lyot latents (reference / science window); None = ote_guidance_scale

# Star flux: testing.py only rescales it to the data in stages 1.1 and 2.1 (hardcoded_match_flux, counts in the central 30 px);
# fit_flux also fits a relative flux scale by gradient (priors/grids.py FluxScale; own optimizer group, lr flux_lr = relative step)
# in every other stage, so that the model can dim the halo without scattering light out of the field with high-frequency OPD
fit_flux = False
flux_lr = 1e-3

# Logging
log_opd_every = 50 # high/low frequency content of the OTE OPD and of the Lyot-plane (NIRCam) OPD
snapshot_opd_every = 500 # full OTE and Lyot-plane OPD maps
state_iters = [0, 600, 2600, 4100, 5500, 8100, 11000] # full model state saved at these iterations (fields are recomputed from them by priors/plots.py)

# Injected planet (position in px from the array center, flux relative to the 5 sigma noise limit) and the real planet
# HIP 65426 b: its own SNR is tracked as in run_FLIP.py (blob of rad_arcsec), and a disk of mask_rad_arcsec around it is masked
# out of the injected planet's SNR (run_FLIP.py: 1.3", which would cover an injection at (10, 0) px, so 0.5" by default)
inj_pos_px, inj_n_5sigma = (10., 0.), 5
real_planet = {'pos_x_px': -7.2, 'pos_y_px': 11., 'rad_arcsec': 1.3, 'mask_rad_arcsec': 0.5}

# Data: 'calints' (testing.py: FITS calints, nothing subtracted) or 'olddemo_npy' (the frames of the original pyFLIP demo,
# run_JWST_test_HIP65426.sh: background-subtracted COADDALL frames and its photon-noise map), for checking against it
data_source = 'calints'
olddemo_dir = '/srv/tmp/droyo/exoplanet/pyflip/HIP65426/demo_data'

# Run variants: overrides of the settings above, selected by the command line argument
olddemo_injection = dict(inj_pos_px=(17., 5.), inj_n_5sigma=6, real_planet={'pos_x_px': -7.2, 'pos_y_px': 11., 'rad_arcsec': 1.3, 'mask_rad_arcsec': 1.3})
VARIANTS = {
    'noprior': dict(USE_PRIOR=False),
    'prior': dict(),
    'prior_gs_hi': dict(ote_guidance_scale=1.5e-2),
    'prior_gs_lo': dict(ote_guidance_scale=1.5e-3),
    'prior_ode': dict(ote_prior_sde=False),
    'prior_long': dict(ref_ote_prior_end=5500),
    'prior_v3sched': dict(ote_prior_sigma_max=1., ote_prior_sigma_min=0.005),
    # verification against the original pyFLIP demo (injection (17, 5) px at 30 x 1 sigma = 6 x 5 sigma)
    'noprior_olddemo_npy': dict(USE_PRIOR=False, data_source='olddemo_npy', **olddemo_injection),
    'noprior_olddemo': dict(USE_PRIOR=False, **olddemo_injection),
    'prior_olddemo': dict(**olddemo_injection),
    'prior_ode_olddemo': dict(ote_prior_sde=False, **olddemo_injection),
    # round two (testing.py injection): mostly ODE prior variants. 'long' = diffusion over the whole reference OTE window
    # (600-5500) and the residual grid never fitted (frozen for the whole run)
    'prior_ode_long': dict(ote_prior_sde=False, ref_ote_prior_end=5500, ote_grid_frozen_until=11001),
    'prior_long': dict(ref_ote_prior_end=5500, ote_grid_frozen_until=11001),
    'prior_ode_end4000': dict(ote_prior_sde=False, ref_ote_prior_end=4000),
    'prior_ode_gridlr': dict(ote_prior_sde=False, ote_grid_lr=1e-9),
    'prior_ode_gs_hi': dict(ote_prior_sde=False, ote_guidance_scale=1.5e-2),
    'prior_ode_v3sched': dict(ote_prior_sde=False, ote_prior_sigma_max=1., ote_prior_sigma_min=0.005),
    'prior_ode_long_gs_hi': dict(ote_prior_sde=False, ref_ote_prior_end=5500, ote_grid_frozen_until=11001, ote_guidance_scale=1.5e-2),
    'prior_ode_seed1': dict(ote_prior_sde=False, ote_prior_seed=1),
    # drift prior: mode 'offset' (drift on the WFS map) on the reference stage, optionally a second DPS window on the science
    # frame (6100-8100, after the position fit; the grid is refined 8100-11000); and the map prior + a science drift window
    'drift_offset': dict(ote_prior_mode='offset'),
    'drift_offset_ode': dict(ote_prior_mode='offset', ote_prior_sde=False),
    'drift_offset_sci': dict(ote_prior_mode='offset', sci_ote_prior_start=6100, sci_ote_prior_end=8100),
    'drift_offset_sci_ode': dict(ote_prior_mode='offset', ote_prior_sde=False, sci_ote_prior_start=6100, sci_ote_prior_end=8100),
    'prior_sci_drift': dict(sci_ote_prior_start=6100, sci_ote_prior_end=8100),
    'prior_ode_sci_drift': dict(ote_prior_sde=False, sci_ote_prior_start=6100, sci_ote_prior_end=8100),
    # round three: drift prior taking over more of the pipeline, reference DPS over the whole OTE window (600-5500, grid never
    # fitted) or 600-2600 (grid 2600-5500, as the other runs); science DPS 6100-8100 with/without the grid refined afterwards
    'drift_offset_long': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001),
    'drift_offset_long_sci': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, sci_ote_prior_start=6100, sci_ote_prior_end=8100, ote_fit_grid_in_science=False),
    'drift_offset_sci_nogrid': dict(ote_prior_mode='offset', sci_ote_prior_start=6100, sci_ote_prior_end=8100, ote_fit_grid_in_science=False),
    'prior_long_sci_drift': dict(ref_ote_prior_end=5500, ote_grid_frozen_until=11001, sci_ote_prior_start=6100, sci_ote_prior_end=8100, ote_fit_grid_in_science=False),
    # round four: Lyot-plane OPD regularization and flux refit, OTE as drift_offset_long (drift prior 600-5500, OTE grid never fitted)
    'noprior_flux': dict(USE_PRIOR=False, fit_flux=True),
    'lyot_frozen': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='frozen'),
    'lyot_frozen_flux': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='frozen', fit_flux=True),
    'lyot_coarse': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='coarse'),
    'lyot_coarse_flux': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='coarse', fit_flux=True),
    'lyot_drift': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift'),
    'lyot_drift_flux': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True),
    'lyot_coarse_flux_sci': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='coarse', fit_flux=True,
                                 sci_ote_prior_start=6100, sci_ote_prior_end=8100, ote_fit_grid_in_science=False),
    # round five: strictly band-limited (Fourier-upsampled) coarse Lyot grids with the flux refit, and lower Lyot grid learning
    # rates (free and coarse); OTE as drift_offset_long. 24 nodes = 12 cycles per array, 32 = 16 (the HF metric starts at 18)
    'lyot_four24_flux': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='coarse', lyot_coarse_upsample='fourier', fit_flux=True),
    'lyot_four32_flux': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='coarse', lyot_coarse_upsample='fourier', lyot_coarse_nodes=32, fit_flux=True),
    'lyot_four32_flux_lr1': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='coarse', lyot_coarse_upsample='fourier', lyot_coarse_nodes=32, lyot_grid_lr=1e-9, fit_flux=True),
    'lyot_free_lr1_flux': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_grid_lr=1e-9, fit_flux=True),
    # round five, second half: the Lyot drift prior given something to adapt on the science frame (its own science DPS window
    # 6100-8100, the OTE science window, or both) and a higher guidance scale; all with the flux refit
    'lyot_drift2_flux': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, lyot_sci_prior_start=6100, lyot_sci_prior_end=8100),
    'lyot_drift_flux_sci': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, sci_ote_prior_start=6100, sci_ote_prior_end=8100, ote_fit_grid_in_science=False),
    'lyot_drift2_flux_sci': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, lyot_sci_prior_start=6100, lyot_sci_prior_end=8100, sci_ote_prior_start=6100, sci_ote_prior_end=8100, ote_fit_grid_in_science=False),
    'lyot_drift_gshi_flux': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, ote_guidance_scale=1.5e-2),
    # round six: Lyot plane static on the reference frame and free (full rate or 1e-9) on the science frame only; Fourier grids
    # (32 and 36 nodes = 16 and 18 cycles per array) with the OTE science drift window. All with the OTE drift prior and the flux refit
    'lyot_scifree_flux': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_grid_frozen_until=6100, fit_flux=True),
    'lyot_scifree_lr1_flux': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_grid_frozen_until=6100, lyot_grid_lr=1e-9, fit_flux=True),
    'lyot_four32_flux_sci': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='coarse', lyot_coarse_upsample='fourier', lyot_coarse_nodes=32, fit_flux=True,
                                 sci_ote_prior_start=6100, sci_ote_prior_end=8100, ote_fit_grid_in_science=False),
    'lyot_four36_flux_sci': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='coarse', lyot_coarse_upsample='fourier', lyot_coarse_nodes=36, fit_flux=True,
                                 sci_ote_prior_start=6100, sci_ote_prior_end=8100, ote_fit_grid_in_science=False),
    # round six, second half: drift Lyot prior with its science window over the whole science stage (6100-11000), with/without
    # the OTE science window (2000 or 4900 steps), and a seed repeat of lyot_drift2_flux_sci for the run-to-run spread
    'lyot_drift2long_flux': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, lyot_sci_prior_start=6100, lyot_sci_prior_end=11000),
    'lyot_drift2long_flux_sci': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, lyot_sci_prior_start=6100, lyot_sci_prior_end=11000, sci_ote_prior_start=6100, sci_ote_prior_end=8100, ote_fit_grid_in_science=False),
    'lyot_drift2long_flux_scilong': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, lyot_sci_prior_start=6100, lyot_sci_prior_end=11000, sci_ote_prior_start=6100, sci_ote_prior_end=11000, ote_fit_grid_in_science=False),
    # round seven: one-knob variations of lyot_drift2long_flux (w_)
    'w_lyot_gs_hi': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, lyot_sci_prior_start=6100, lyot_sci_prior_end=11000, lyot_guidance_scale=1.5e-2),
    'w_lyot_gs_lo': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, lyot_sci_prior_start=6100, lyot_sci_prior_end=11000, lyot_guidance_scale=1.5e-3),
    'w_sci_gs_hi': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, lyot_sci_prior_start=6100, lyot_sci_prior_end=11000, lyot_sci_guidance_scale=1.5e-2),
    'w_ode': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, lyot_sci_prior_start=6100, lyot_sci_prior_end=11000, ote_prior_sde=False),
    'w_noflux': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=False, lyot_sci_prior_start=6100, lyot_sci_prior_end=11000),
    'w_ote_sci_long': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, lyot_sci_prior_start=6100, lyot_sci_prior_end=11000, sci_ote_prior_start=6100, sci_ote_prior_end=11000, ote_fit_grid_in_science=False),
    'w_seed1': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, lyot_sci_prior_start=6100, lyot_sci_prior_end=11000, ote_prior_seed=1),
    'w_lyot_ref_short': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, lyot_sci_prior_start=6100, lyot_sci_prior_end=11000, lyot_prior_end=3500),
    'lyot_drift2_flux_sci_seed1': dict(ote_prior_mode='offset', ref_ote_prior_end=5500, ote_grid_frozen_until=11001, lyot_mode='drift', fit_flux=True, lyot_sci_prior_start=6100, lyot_sci_prior_end=8100, sci_ote_prior_start=6100, sci_ote_prior_end=8100, ote_fit_grid_in_science=False, ote_prior_seed=1),
}
variant = sys.argv[1]
globals().update(VARIANTS[variant])
np.random.seed(0) # same injected planet flux in every variant (the injection draws a noise realization)


#### MODEL GENERATION


peak_flux_star  = nn.Parameter(torch.FloatTensor([1.]))

wf_npix = 1024
diameter = 6.603464
DEVICE = 'cuda'
num_wl = 21
oversample = 2
data_dir = priors_dir + '/model_setup'
out_dir = '/srv/tmp/droyo/exoplanet/pyflip-nw/outputs/' + variant
os.makedirs(out_dir, exist_ok=True)
num_det_px = 80

# Stellar position
offset_STAR = nn.Parameter(torch.FloatTensor([0., 0.])).to(DEVICE)

# TODO: wrapper function to do all of this at once for the simpler cases, including using STPSF to generate the masks and such
_, _, _, _, wlen_weights, _ = load_model_data(data_dir=data_dir)

# Set up the wavefront object: this will be propagated by our optical model
wavefront_broad = BroadbandWavefront(wf_npix, diameter, wlen_weights, peak_flux_star, offset_STAR).to(DEVICE)
if fit_flux:
    wavefront_broad.flux_correction = FluxScale(DEVICE=DEVICE)

diffnircam = assemble_JWST_NIRCam_coron(data_dir,num_det_px)

if USE_PRIOR:
    # Replace the free OTE OPD grid (OPDOffsetModule) of the primary by the diffusion prior. It keeps a 'grid' parameter,
    # so the optimizer groups and default_stages below work unchanged.
    field_term = None
    if ote_prior_mode == 'replace':
        field_term = torch.from_numpy(np.flip(np.load(ote_field_term_path), axis=0).copy())[None] # flipped like the WFS map in load_model_data
    ref_prior = OPDPrior(**(ote_map_prior if ote_prior_mode == 'replace' else ote_drift_prior), DEVICE=DEVICE)
    phases = [DPSPhase(ref_prior, ref_ote_prior_start, ref_ote_prior_end, sigma_max=ote_prior_sigma_max, sigma_min=ote_prior_sigma_min, sde=ote_prior_sde, seed=ote_prior_seed, DEVICE=DEVICE)]
    if sci_ote_prior_start is not None:
        sci_prior = ref_prior if ref_prior.kind == 'drift' else OPDPrior(**ote_drift_prior, DEVICE=DEVICE)
        phases.append(DPSPhase(sci_prior, sci_ote_prior_start, sci_ote_prior_end, sigma_max=ote_prior_sigma_max, sigma_min=ote_prior_sigma_min, sde=ote_prior_sde,
                               seed=None if ote_prior_seed is None else ote_prior_seed + 1, DEVICE=DEVICE))
    ote_prior = DPSAdamOPD(phases, mode=ote_prior_mode, field_term=field_term,
                           grid_frozen_until=ote_grid_frozen_until, DEVICE=DEVICE)
    diffnircam.optical_system.JWSTPrimary.wfe_offsets = ote_prior

# Lyot-plane OPD treatment (the module keeps a 'grid' parameter, as the OTE one, so the optimizer groups and stages work unchanged)
lyot_prior = None
if lyot_mode == 'coarse':
    diffnircam.optical_system.NIRCamLyotstop.wfe_offsets = CoarseOPDGrid(lyot_coarse_nodes, wf_npix, upsample=lyot_coarse_upsample, DEVICE=DEVICE)
elif lyot_mode == 'drift':
    drift_prior = ote_prior.phases[0].prior if USE_PRIOR and ote_prior.phases[0].prior.kind == 'drift' else OPDPrior(**ote_drift_prior, DEVICE=DEVICE)
    lyot_phases = [DPSPhase(drift_prior, lyot_prior_start, lyot_prior_end, sigma_max=ote_prior_sigma_max, sigma_min=ote_prior_sigma_min, sde=ote_prior_sde,
                            seed=None if ote_prior_seed is None else ote_prior_seed + 2, DEVICE=DEVICE)]
    if lyot_sci_prior_start is not None:
        lyot_phases.append(DPSPhase(drift_prior, lyot_sci_prior_start, lyot_sci_prior_end, sigma_max=ote_prior_sigma_max, sigma_min=ote_prior_sigma_min, sde=ote_prior_sde,
                                    seed=None if ote_prior_seed is None else ote_prior_seed + 3, DEVICE=DEVICE))
    lyot_prior = DPSAdamOPD(lyot_phases, mode='offset', grid_frozen_until=11001, DEVICE=DEVICE)
    diffnircam.optical_system.NIRCamLyotstop.wfe_offsets = lyot_prior
lyot_grid = diffnircam.optical_system.NIRCamLyotstop.wfe_offsets.grid

result = diffnircam(wavefront_broad)

result = result.detach().cpu().numpy()

np.save(out_dir + '/res2.npy', result)


#### OPTIMIZER SETUP

scheduler_iters = 20000 # Max number of iterations for LR scheduler: we can halt the optimization before but not after this

# Choose parameters that are going to be optimized, and assign them reasonable learning rates

# OPD in priamry and post Lyot stop
opds = [
    diffnircam.optical_system.JWSTPrimary.wfe_offsets.grid,
    diffnircam.optical_system.NIRCamLyotstop.wfe_offsets.grid
]

# Star position offset from coronagraph center
star_position = [
    wavefront_broad.angles_offset
]

# Detector flats 
detector_flats = [
    diffnircam.optical_system.NIRCamdetector.detector_flat_field.grid
]

# The 4 works well for many datasets but could be tweaked in some cases; e.g. we have found it needs to be
# smaller for fainter stars sometimes
optimizer_argument = [
        {'params': opds, 'lr': 4e-9},
        {'params': star_position, 'lr': 4e-8},
        {'params': detector_flats, 'lr': 4e-6}
    ]
if USE_PRIOR:
    optimizer_argument.append(ote_prior.param_group(ote_guidance_scale)) # The likelihood step of DPS-Adam is this optimizer's step on the latent
    optimizer_argument[0]['params'] = [p for p in opds if p is not ote_prior.grid] # The residual OTE grid in its own group (same as testing.py when ote_grid_lr = 4e-9)
    optimizer_argument.append({'params': [ote_prior.grid], 'lr': ote_grid_lr})
if lyot_mode == 'frozen':
    optimizer_argument[0]['params'] = [p for p in optimizer_argument[0]['params'] if p is not lyot_grid] # Lyot grid never updated
if lyot_prior is not None:
    lyot_scales = [ote_guidance_scale if lyot_guidance_scale is None else lyot_guidance_scale, ote_guidance_scale if lyot_sci_guidance_scale is None else lyot_sci_guidance_scale]
    optimizer_argument += [{'params': [lat], 'lr': lr} for lat, lr in zip(lyot_prior.latents, lyot_scales)] # one group per Lyot DPS window
if lyot_grid_lr is not None:
    optimizer_argument[0]['params'] = [p for p in optimizer_argument[0]['params'] if p is not lyot_grid]
    optimizer_argument.append({'params': [lyot_grid], 'lr': lyot_grid_lr}) # Lyot grid in its own group (stages/hooks act on .grad, unchanged)
if fit_flux:
    optimizer_argument.append({'params': [wavefront_broad.flux_correction.log_scale], 'lr': flux_lr})

optimizer = torch.optim.AdamW(optimizer_argument, weight_decay=0.0)

# Make eta_min the same as OPDs LR, we just find it does well, but we might want something different at some point
# because making all parameters have the same LR as the minimum limit may be sub-optimal

# TODO: we might want to get rid fo the scheduler altogether at some point, or 
# test how the eta_min affects parameters with different learning rates
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_iters, eta_min=4e-9) 


######### LOAD RELEVANT DATA
refdatapath = calints_dir + '/jw01386001001_0310e_00001_nrcalong_calints.fits'
scidatapath = calints_dir + '/jw01386002001_0310a_00001_nrcalong_calints.fits'
if data_source == 'calints':
    ref, ref_noise = load_and_crop_data(refdatapath, num_det_px=80, coron_type='MASK335R')
    sci, sci_noise = load_and_crop_data(scidatapath, num_det_px=80, coron_type='MASK335R')
else:
    crop = lambda a: a[0, 134:214, 110:190][None].astype(np.float32) # the crop of load_and_crop_data for MASK335R (320x320 subarray)
    ref = crop(np.load(olddemo_dir + '/HIP-68245/full_im/reference_001_00001COADDALL.npy'))
    sci = crop(np.load(olddemo_dir + '/HIP-65426/full_im/target_002_00001COADDALL.npy'))
    sci_noise = crop(np.load(olddemo_dir + '/HIP-65426/noise_maps/noisemap_target_002_00001COADDALL.npy'))

# Array with 0s and 1s, used to mask certain pixels (e.g., dead or hot pixels) from contributing to the loss function
# Has to be provided by the user, probably after a first pass. Will depend on the pixel cleaning routine used.
# It can also be used to mask out full regions of the image from the optimization, for arbitrary testing.
px_mask = np.load(priors_dir + '/pixel_masks.npy')
px_mask = px_mask.astype(np.float32)
px_mask_numpy = px_mask.copy()
px_mask = torch.from_numpy(px_mask).to(DEVICE)[None]

### Inject planet(s) and such to track SNR
# Only needed on science target really

flux_n_5_sigma = inj_n_5sigma # inject planet N times brighter than the 5 sigma photon noise limit

fakes = [{'pos_x_px': inj_pos_px[0], 'pos_y_px': inj_pos_px[1]}] # fake planet locations, relative to center of the array, in px

sci, injected_fluxes = inject_fake_planets(sci,sci_noise, wavefront_broad, diffnircam,fakes, flux_n_5_sigma, smooth_sigma_for_flux=1.)

ref = torch.from_numpy(ref).to(DEVICE)
sci = torch.from_numpy(sci).to(DEVICE)

####### RUN OPTIMIZATION
sci_loss_weight = 0.005 # How much to weigh down the loss function when evaluating on science target relative to reference target; essentially make optimization slower

measure_SNR_every = 100 # measure the SNR of the fake planet(s) every how many iterations 

stage_cutoffs = [600, 3500, 5500, 6100, 11000] # Iterations separating stage 1.1, 1.2, 1.3, 2.1 and 2.2 in our default stage framework


tbar = tqdm.tqdm(range(scheduler_iters + 1), mininterval=30.0)

# Track injected planets
signal_peaks_arr, signal_losses_arr, snrs_arr, track_iters = [], [], [], []
real_snrs = [] # SNR of the real planet, as run_FLIP.py: blob of rad_arcsec, annulus width 0.5", smoothed with 1 px
# Two SNR definitions: track_SNR takes the signal from the unsmoothed residual and the noise from the 1 px smoothed one;
# run_FLIP.py (blur_before_signal) takes both from the smoothed residual (about 1.65x lower). Both are logged.
snrs_smooth_arr, real_snrs_smooth, real_peaks = [], [], [] # real_peaks: recovered peak of the real planet (MJy/sr), no true flux is known
psf_pixel_scale_arcsec = 0.062424185
if real_planet is not None:
    real_x, real_y = real_planet['pos_x_px'] * psf_pixel_scale_arcsec, real_planet['pos_y_px'] * psf_pixel_scale_arcsec

# Track the OTE OPD and the Lyot-plane (NIRCam) OPD layers
pupil = diffnircam.optical_system.JWSTPrimary.amplitude[0].cpu().numpy() > 0
lyot_pupil = diffnircam.optical_system.NIRCamLyotstop.amplitude[0].cpu().numpy() > 0
central_wl = num_wl // 2
opd_iters, opd_hf_rms, opd_lf_rms, opd_snap_iters, opd_snaps, loss_iters, losses = [], [], [], [], [], [], []
lyot_hf_rms, lyot_lf_rms, lyot_snaps = [], [], []
flux_factors, flux_ratios = [], [] # fitted flux factor (1 = the hardcoded flux matching) and model / data counts over the pixel mask
def current_ote_opd():
    primary = diffnircam.optical_system.JWSTPrimary
    with torch.no_grad():
        return primary.wfe_offsets(primary.opd)[0].detach().cpu().numpy().astype(np.float64)
def current_lyot_opd():
    # NIRCam OPD at the central wavelength + the free Lyot-plane grid
    lyot = diffnircam.optical_system.NIRCamLyotstop
    with torch.no_grad():
        return lyot.wfe_offsets(lyot.opd)[0, central_wl].detach().cpu().numpy().astype(np.float64)
os.makedirs(out_dir + '/states', exist_ok=True)
def save_state(i):
    # Everything needed to re-run the forward model at iteration i offline (priors/plots.py): the parameters of the optical
    # system and the wavefront, plus the OTE OPD layer (the prior parts of DPSAdamOPD are not parameters)
    cpu = lambda sd: {k: v.detach().cpu() for k, v in sd.items()}
    wf_state = cpu(wavefront_broad.state_dict())
    if fit_flux: # fold the fitted scale into the plain flux factor, as a FluxOffsetModule would hold it
        wf_state['flux_correction.data'] = wf_state['flux_correction.data'] * wf_state.pop('flux_correction.log_scale').exp()
    torch.save({'iteration': i, 'model': cpu(diffnircam.state_dict()), 'wavefront': wf_state,
                'ote_opd': torch.from_numpy(current_ote_opd()).float()}, f'{out_dir}/states/iter{i}.pt')

for i in tbar:

    # Re start optimizer at the start always (IMPORTANT!)
    optimizer.zero_grad()

    if i == 0 and 0 in state_iters:
        save_state(0)

    if USE_PRIOR:
        ote_prior.step(i) # Prior move of the latent (previous diffusion step) and denoised OTE OPD for this iteration
    if lyot_prior is not None:
        lyot_prior.step(i)

    # Model new PSF
    PSF = diffnircam(wavefront_broad)

    if i < stage_cutoffs[2]: # Iteration dividing stage 1.3 from 2.1: go from calibration on reference to fitting science
        loss = F.smooth_l1_loss(PSF, ref, reduction='none') 
    else:
        loss = F.l1_loss(PSF, sci, reduction='none') * sci_loss_weight

    loss *= px_mask # Mask out bad pixels so they don't contribute to loss function
    loss = loss.sum() / px_mask.sum() # Collapse loss into a number

    # At this step, we can add some regularization to the loss function if we want. E.g., smoothness or whatever
    # Otherwise, continue to backpropagation and parameter update

    # Backpropagate loss function, populate parameter gradients
    loss.backward()

    if i < stage_cutoffs[0]: # TODO: fix flux fitting with gradient; then this will not be necessary
        wavefront_broad = hardcoded_match_flux(PSF,ref,px_mask,wavefront_broad)
    elif i > stage_cutoffs[2] and i < stage_cutoffs[3]:
        wavefront_broad = hardcoded_match_flux(PSF,sci,px_mask,wavefront_broad)
    if fit_flux and (i <= stage_cutoffs[0] or stage_cutoffs[2] <= i <= stage_cutoffs[3]):
        wavefront_broad.flux_correction.log_scale.grad *= 0. # flux by gradient only in the stages without the hardcoded rescaling

    if lyot_grid_frozen_until is not None and i < lyot_grid_frozen_until:
        lyot_grid.grad *= 0.
    if USE_PRIOR:
        prior_stages(diffnircam, wavefront_broad, i, stage_cutoffs=stage_cutoffs, fit_ote_in_science=(sci_ote_prior_start is not None) if ote_fit_grid_in_science is None else ote_fit_grid_in_science) # default_stages, plus the OTE grid refined in stage 2.2 after a science DPS window
    else:
        default_stages(diffnircam, wavefront_broad, i, stage_cutoffs=stage_cutoffs) # Zero-out (in-place) gradients of parameters we are not fitting at a given stage

    # Update parameters 
    optimizer.step()
    scheduler.step()

    if i % 10 == 0:
        loss_iters.append(i); losses.append(loss.item())
    if i % log_opd_every == 0:
        opd_now = current_ote_opd()
        low, high = opd_bandpass(opd_now, pupil)
        opd_iters.append(i); opd_hf_rms.append(opd_rms_nm(high, pupil)); opd_lf_rms.append(opd_rms_nm(low, pupil))
        low, high = opd_bandpass(current_lyot_opd(), lyot_pupil)
        lyot_hf_rms.append(opd_rms_nm(high, lyot_pupil)); lyot_lf_rms.append(opd_rms_nm(low, lyot_pupil))
        target = ref if i < stage_cutoffs[2] else sci
        flux_factors.append(wavefront_broad.flux_correction.factor().item() if fit_flux else 1.); flux_ratios.append(((PSF * px_mask).sum() / (target * px_mask).sum()).item())
    if i % snapshot_opd_every == 0:
        opd_snap_iters.append(i); opd_snaps.append(current_ote_opd().astype(np.float32)); lyot_snaps.append(current_lyot_opd().astype(np.float32))
    if i + 1 in state_iters:
        save_state(i + 1) # state after i + 1 updates = the state used by iteration i + 1

    if i == stage_cutoffs[2] -1:
        # End of calibration!
        residual_ref = ref - PSF 
        residual_ref = residual_ref.detach().cpu().numpy()
        np.save(out_dir + '/residual_ref.npy', residual_ref)
        residual_ref_final = residual_ref

    if i == stage_cutoffs[-1]:
        # End! 
        residual_sci = sci - PSF 
        residual_sci = residual_sci.detach().cpu().numpy()
        np.save(out_dir + '/residual_sci.npy', residual_sci)
        break
        


    # Now, track SNR of injected planet(s)
    if i%measure_SNR_every == 0 and i > stage_cutoffs[2]: # only measure in science target
        residual_sci = sci - PSF 
        residual_sci = residual_sci.detach().cpu().numpy()
        residual_for_fakes = residual_sci
        if real_planet is not None:
            residual_for_fakes = circular_mask(residual_sci[0], real_x, real_y, real_planet['mask_rad_arcsec'], np.nan)[None] # mask the real planet out
            px_mask_nanned = np.where(px_mask_numpy == 0., np.nan, px_mask_numpy)
            real_snr, _, _ = calc_snr(residual_sci[0] * px_mask_nanned, real_x, real_y, real_planet['rad_arcsec'], np.sqrt(real_x**2 + real_y**2), width=0.5, sigma_kernel=1.)
            real_snrs.append(real_snr)
            real_snr_s, _, _, real_peak = calc_snr(residual_sci[0] * px_mask_nanned, real_x, real_y, real_planet['rad_arcsec'], np.sqrt(real_x**2 + real_y**2), width=0.5, sigma_kernel=1., blur_before_signal=True, return_signal_peak=True)
            real_snrs_smooth.append(real_snr_s); real_peaks.append(real_peak)
        px_mask_nanned = np.where(px_mask_numpy == 0., np.nan, px_mask_numpy)
        snrs_smooth_arr.append([calc_snr(residual_for_fakes[0] * px_mask_nanned, f['pos_x_px'] * psf_pixel_scale_arcsec, f['pos_y_px'] * psf_pixel_scale_arcsec, 0.3,
                                         np.sqrt(f['pos_x_px']**2 + f['pos_y_px']**2) * psf_pixel_scale_arcsec, sigma_kernel=1., blur_before_signal=True)[0] for f in fakes])
        signal_peaks, signal_losses, snrs = track_SNR(residual_for_fakes, i, fakes, injected_fluxes,mask_name='MASK335R', snr_smooth_sigma=1., px_mask_numpy=px_mask_numpy, save_plot=False)
        signal_peaks_arr.append(signal_peaks)
        signal_losses_arr.append(signal_losses)
        snrs_arr.append(snrs)
        track_iters.append(i)

# After everything, save these diagnostics and such 
signal_peaks_arr = np.array(signal_peaks_arr)
signal_losses_arr = np.array(signal_losses_arr)
snrs_arr = np.array(snrs_arr)
track_iters = np.array(track_iters)

np.savez(out_dir + '/metrics.npz', variant=variant, settings=str(VARIANTS[variant]), stage_cutoffs=np.array(stage_cutoffs),
         ote_prior_start=ref_ote_prior_start, ote_prior_end=ref_ote_prior_end, sci_ote_prior_start=np.array(sci_ote_prior_start, dtype=float), sci_ote_prior_end=np.array(sci_ote_prior_end, dtype=float),
         loss_iters=np.array(loss_iters), losses=np.array(losses),
         opd_iters=np.array(opd_iters), opd_hf_rms_nm=np.array(opd_hf_rms), opd_lf_rms_nm=np.array(opd_lf_rms),
         opd_snap_iters=np.array(opd_snap_iters), opd_snaps=np.array(opd_snaps), opd_final=current_ote_opd().astype(np.float32), pupil=pupil,
         flux_factors=np.array(flux_factors), flux_ratios=np.array(flux_ratios), lyot_mode=lyot_mode, fit_flux=fit_flux,
         lyot_hf_rms_nm=np.array(lyot_hf_rms), lyot_lf_rms_nm=np.array(lyot_lf_rms), lyot_snaps=np.array(lyot_snaps), lyot_final=current_lyot_opd().astype(np.float32), lyot_pupil=lyot_pupil,
         state_iters=np.array(sorted(state_iters)), data_dir=data_dir, num_det_px=num_det_px, wf_npix=wf_npix, diameter=diameter,
         track_iters=track_iters, snrs=snrs_arr, signal_peaks=signal_peaks_arr, signal_losses=signal_losses_arr,
         snrs_smooth=np.array(snrs_smooth_arr), real_snrs_smooth=np.array(real_snrs_smooth), real_peaks=np.array(real_peaks),
         real_snrs=np.array(real_snrs), real_planet_px=np.array([real_planet['pos_x_px'], real_planet['pos_y_px']]) if real_planet is not None else np.zeros(0),
         fakes_px=np.array([[f['pos_x_px'], f['pos_y_px']] for f in fakes]), injected_fluxes=np.array(injected_fluxes),
         residual_ref=residual_ref_final, residual_sci=residual_sci, sci=sci.cpu().numpy(), ref=ref.cpu().numpy(), px_mask=px_mask_numpy)
print('saved', out_dir + '/metrics.npz')