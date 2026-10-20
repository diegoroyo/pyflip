"""testing.py with the Babinet occulter, the FPM axial offset and the Lyot stop
shift and rotation fitted on the reference star. Paths and options below, or from
the environment variables named there. Run from refactoring/: the modules come from
there, and inject_fake_planets reads data/ relative to the working directory.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'refactoring'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from wavefronts import BroadbandWavefront
from model_building import assemble_JWST_NIRCam_coron
from optimization_handlers import default_stages
from dl_utils import load_model_data, load_and_crop_data, inject_fake_planets, hardcoded_match_flux, track_SNR

#### OPTIONS
data_dir = os.environ.get('MODEL_DIR', '/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/uuu')
refdatapath = os.environ.get('REF_FILE', '/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/HIP65426_newmodel/NEWGOODDATA/onlycalints_nanreplaced/jw01386001001_0310e_00001_nrcalong_calints.fits')
scidatapath = os.environ.get('SCI_FILE', '/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/HIP65426_newmodel/NEWGOODDATA/onlycalints_nanreplaced/jw01386002001_0310a_00001_nrcalong_calints.fits')
px_mask_path = os.environ.get('PX_MASK', '/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/HIP65426/model_setup/pixel_masks/pixel_masks.npy')
out_dir = os.environ.get('OUT_DIR', 'geometry_fit')
fpm_raster = os.environ.get('FPM_RASTER')                    # None: smooth mask; else the real mask's raster (.npz)
um_per_arcsec = float(os.environ.get('UM_PER_ARCSEC', 593.336))  # real mask scale on the sky [um/arcsec]
mount = os.environ.get('MOUNT', '1') == '1'                  # keep the mask mount regions
fpm_focal_length = 131.4                                     # equivalent focal length [m]
stage_cutoffs = [int(c) for c in os.environ.get('STAGES', '600,3500,5500,6100,11000').split(',')]
scheduler_iters = 20000

DEVICE = os.environ.get('DEVICE', 'cuda')
wf_npix = 1024
diameter = 6.603464
num_det_px = 80
os.makedirs(out_dir, exist_ok=True)

#### MODEL
peak_flux_star = nn.Parameter(torch.FloatTensor([1.]))
offset_STAR = nn.Parameter(torch.FloatTensor([0., 0.])).to(DEVICE)
_, _, _, _, wlen_weights, _ = load_model_data(data_dir=data_dir)
wavefront_broad = BroadbandWavefront(wf_npix, diameter, wlen_weights, peak_flux_star, offset_STAR, DEVICE=DEVICE).to(DEVICE)

diffnircam = assemble_JWST_NIRCam_coron(data_dir, num_det_px, DEVICE=DEVICE, babinet=True, mount=mount, fpm_raster=fpm_raster,
                                        um_per_arcsec=um_per_arcsec, fpm_axial_offset=0., fpm_focal_length=fpm_focal_length)
fpm = diffnircam.optical_system.NIRCamcoronfocalplanemask
lyot = diffnircam.optical_system.NIRCamLyotstop

#### OPTIMIZER
opds = [diffnircam.optical_system.JWSTPrimary.wfe_offsets.grid, lyot.wfe_offsets.grid]
star_position = [wavefront_broad.angles_offset]
detector_flats = [diffnircam.optical_system.NIRCamdetector.detector_flat_field.grid]
lyot_alignment = [lyot.pupil_shifts.x_shift, lyot.pupil_shifts.y_shift, lyot.pupil_shifts.rotation]   # [samples], [samples], [deg]
optimizer_argument = [
        {'params': opds, 'lr': 4e-9},
        {'params': star_position, 'lr': 4e-8},
        {'params': detector_flats, 'lr': 4e-6},
        {'params': [fpm.axial_offset], 'lr': 1e-4},          # [primary-equivalent m]
        {'params': lyot_alignment, 'lr': 1e-3},
    ]
optimizer = torch.optim.AdamW(optimizer_argument, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_iters, eta_min=4e-9)

#### DATA
ref, ref_noise = load_and_crop_data(refdatapath, num_det_px=num_det_px, coron_type='MASK335R')
sci, sci_noise = load_and_crop_data(scidatapath, num_det_px=num_det_px, coron_type='MASK335R')
px_mask_numpy = np.load(px_mask_path).astype(np.float32)
px_mask = torch.from_numpy(px_mask_numpy).to(DEVICE)[None]

fakes = [{'pos_x_px': 10., 'pos_y_px': 0.}]                   # fake planet, relative to the array centre [px]
sci, injected_fluxes = inject_fake_planets(sci, sci_noise, wavefront_broad, diffnircam, fakes, 5, smooth_sigma_for_flux=1.)
ref = torch.from_numpy(ref).to(DEVICE)
sci = torch.from_numpy(sci).to(DEVICE)

#### FIT
sci_loss_weight = 0.005
measure_SNR_every = 300
geometry, snrs_arr, track_iters = [], [], []
for i in tqdm.tqdm(range(scheduler_iters + 1), mininterval=30.0):
    optimizer.zero_grad()
    PSF = diffnircam(wavefront_broad)
    if i < stage_cutoffs[2]:
        loss = F.smooth_l1_loss(PSF, ref, reduction='none')
    else:
        loss = F.l1_loss(PSF, sci, reduction='none') * sci_loss_weight
    loss = (loss * px_mask).sum() / px_mask.sum()
    loss.backward()

    if i < stage_cutoffs[0]:
        wavefront_broad = hardcoded_match_flux(PSF, ref, px_mask, wavefront_broad)
    elif i > stage_cutoffs[2] and i < stage_cutoffs[3]:
        wavefront_broad = hardcoded_match_flux(PSF, sci, px_mask, wavefront_broad)
    default_stages(diffnircam, wavefront_broad, i, stage_cutoffs=stage_cutoffs)
    optimizer.step()
    scheduler.step()

    geometry.append([fpm.axial_offset.item()] + [p.item() for p in lyot_alignment])
    if i == stage_cutoffs[2] - 1:
        np.save(os.path.join(out_dir, 'residual_ref.npy'), (ref - PSF).detach().cpu().numpy())
        print('reference fit: FPM axial offset [m], Lyot shift x, y [samples], rotation [deg]:', geometry[-1])
    if i % measure_SNR_every == 0 and i > stage_cutoffs[2]:
        residual_sci = (sci - PSF).detach().cpu().numpy()
        snrs_arr.append(track_SNR(residual_sci, i, fakes, injected_fluxes, mask_name='MASK335R', snr_smooth_sigma=1.,
                                  px_mask_numpy=px_mask_numpy)[2])
        track_iters.append(i)
    if i == stage_cutoffs[-1]:
        np.save(os.path.join(out_dir, 'residual_sci.npy'), (sci - PSF).detach().cpu().numpy())
        break

np.save(os.path.join(out_dir, 'geometry.npy'), np.array(geometry))
np.save(os.path.join(out_dir, 'snrs.npy'), np.array(snrs_arr))
np.save(os.path.join(out_dir, 'snr_iters.npy'), np.array(track_iters))
print('final geometry: FPM axial offset [m], Lyot shift x, y [samples], rotation [deg]:', geometry[-1])
