"""testing.py's staged fit (reference calibration, then science) with the instrument
geometry held at the values fit_geometry_cases.py fitted on the reference star, and
scores: noise-weighted chi^2 / N on the reference (fitted and held-out frames) and the
science rolls, the real companion's SNR, and the SNR of planets injected into the
held-out reference frames.

CASE: a mask of fit_geometry_cases.py with its fitted geometry and the stop LYOT_FILE;
'_sq' appended treats the band-limited raster's samples as square cells. CASE=flip is
PyFLIP as shipped: its mask file (FFT occulter, with the mount), the model's stop, no
FPM axial offset, its registration and blur. LOSS=chi2 replaces testing.py's smooth L1 /
L1 losses and window flux rule by the noise-weighted chi^2 with the flux solved in closed
form at every iteration. TOTFRAC=f keeps the detector window's share of the light after the
Lyot stop at least f times that of the zero-map model (penalty 35 x the zero-map loss x
relu(f - share ratio)^2); DEADZONE=t [nm] adds DEADLAM x the mean over each map's support of
relu(|map - mean| - t)^2 / t^2. Both are weighted like the stage's data term.

Run from refactoring/ (inject_fake_planets reads data/ relative to the working
directory). Paths and options below, or from the environment variables named there.
Saves the maps, residuals and scores to OUT_DIR and prints the scores. The fitted state at
the end of the calibration and at the end of the fit (maps, flat field, star, flux,
geometry and the options to rebuild the model) is saved as fitted_reference.npz and
fitted_final.npz, the input of null_space_walk.py.
"""
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'refactoring'))
import copy
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
refdatapath = os.environ['REF_FILE']                         # reference frame the model is calibrated on
held_files = os.environ['HELD_FILES'].split(',')             # reference frames scored with the calibrated model
sci_files = os.environ['SCI_FILES'].split(',')               # science rolls; the staged fit uses the first
px_mask_path = os.environ.get('PX_MASK', '/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/HIP65426/model_setup/pixel_masks/pixel_masks.npy')
case = os.environ.get('CASE', 'real593_bl_sq')
geometry_dir = os.environ.get('GEOMETRY_DIR')                # output of fit_geometry_cases.py (not used by CASE=flip)
fpm_raster = os.environ.get('FPM_RASTER')                    # real mask raster (.npz)
lyot_file = os.environ.get('LYOT_FILE')                      # stpsf's MASKRND transmission, 1024 x 1024
out_dir = os.environ.get('OUT_DIR', 'flip_stages_' + case)
loss_kind = os.environ.get('LOSS', 'flip')                   # flip (testing.py) | chi2
total_fraction = float(os.environ.get('TOTFRAC', 0))         # 0: no light constraint
dead_zone = float(os.environ.get('DEADZONE', 0)) * 1e-9       # [m]; 0: no map penalty
dead_lam = float(os.environ.get('DEADLAM', 0.3))
companion = [float(c) for c in os.environ.get('COMPANION', '-8.5,9.5').split(',')]   # in the first roll, from the array centre [px]
fpm_focal_length = 131.4                                     # equivalent focal length [m]
um_per_arcsec = {'593': 593.336, '637': 637.045}
stage_cutoffs = [int(c) for c in os.environ.get('STAGES', '600,3500,5500,6100,11000').split(',')]
scheduler_iters = 20000
star_iters = int(os.environ.get('STAR_ITERS', 200))          # star refits on the held-out and second science frames
injection_seps = [7., 10., 13., 17., 21.]                    # injected planets [px from the array centre]
injection_angles = [0., 90., 180., 270.]                     # [deg]

DEVICE = os.environ.get('DEVICE', 'cuda')
wf_npix = 1024
diameter = 6.603464
num_det_px = int(os.environ.get('NUM_DET_PX', 80))              # detector window [pixels]
oversample = 2
os.makedirs(out_dir, exist_ok=True)
np.random.seed(0)

#### MODEL
peak_flux_star = nn.Parameter(torch.FloatTensor([1.]))
offset_STAR = nn.Parameter(torch.FloatTensor([0., 0.])).to(DEVICE)
_, _, _, _, wlen_weights, _ = load_model_data(data_dir=data_dir)
wavefront_broad = BroadbandWavefront(wf_npix, diameter, wlen_weights, peak_flux_star, offset_STAR, DEVICE=DEVICE).to(DEVICE)

if case == 'flip':
    diffnircam = assemble_JWST_NIRCam_coron(data_dir, num_det_px, DEVICE=DEVICE)
else:
    geometry_file = np.load(os.path.join(geometry_dir, 'geometry_cases.npz'))
    fitted_cases = list(geometry_file['cases'])
    geometry = dict(zip(geometry_file['keys'], geometry_file['values'][fitted_cases.index(case if case in fitted_cases else case.replace('_sq', ''))]))
    mask, scale = case.split('_')[0][:-3], case.split('_')[0][-3:]
    diffnircam = assemble_JWST_NIRCam_coron(data_dir, num_det_px, DEVICE=DEVICE, babinet=True, mount=False,
                                            fpm_raster=os.path.join(geometry_dir, 'smooth_MASK335R_raster.npz') if mask == 'smooth' else fpm_raster,
                                            um_per_arcsec=um_per_arcsec[scale], band_limited=mask == 'smooth' or '_bl' in case,
                                            square_cells=case.endswith('_sq'), fpm_axial_offset=1e-3 * geometry['FPM axial offset [mm]'],
                                            fpm_focal_length=fpm_focal_length)
    lyot = diffnircam.optical_system.NIRCamLyotstop
    lyot.amplitude = torch.FloatTensor(np.load(lyot_file)).to(DEVICE)[None]
    detector = diffnircam.optical_system.NIRCamdetector
    detector.subpixel_shift.shift_x = float(geometry['registration x [px]'])
    detector.subpixel_shift.shift_y = float(geometry['registration y [px]'])
    with torch.no_grad():
        lyot.pupil_shifts.x_shift.fill_(wf_npix * geometry['Lyot shift x [% D]'] / 100)
        lyot.pupil_shifts.y_shift.fill_(wf_npix * geometry['Lyot shift y [% D]'] / 100)
        lyot.pupil_shifts.rotation.fill_(geometry['Lyot clocking [deg]'])
        detector.charge_diffusion.log_sigma.fill_(np.log(geometry['blur [px]'] * oversample))
for parameter in diffnircam.parameters():
    parameter.requires_grad_(False)

#### OPTIMIZER
opds = [diffnircam.optical_system.JWSTPrimary.wfe_offsets.grid, diffnircam.optical_system.NIRCamLyotstop.wfe_offsets.grid]
star_position = [wavefront_broad.angles_offset]
detector_flats = [diffnircam.optical_system.NIRCamdetector.detector_flat_field.grid]
for parameter in opds + star_position + detector_flats:
    parameter.requires_grad_(True)
optimizer = torch.optim.AdamW([{'params': opds, 'lr': 4e-9}, {'params': star_position, 'lr': 4e-8},
                               {'params': detector_flats, 'lr': 4e-6}], weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_iters, eta_min=4e-9)

#### DATA
px_mask_numpy = np.load(px_mask_path).astype(np.float32)
px_mask = torch.from_numpy(px_mask_numpy).to(DEVICE)[None]
ref, ref_noise = load_and_crop_data(refdatapath, num_det_px=num_det_px, coron_type='MASK335R')
sci, sci_noise = load_and_crop_data(sci_files[0], num_det_px=num_det_px, coron_type='MASK335R')
fakes = [{'pos_x_px': 10., 'pos_y_px': 0.}]                   # testing.py's injected planet, relative to the array centre [px]
sci, injected_fluxes = inject_fake_planets(sci, sci_noise, wavefront_broad, diffnircam, fakes, 5, smooth_sigma_for_flux=1., DEVICE=DEVICE)
ref_weights, sci_weights = [torch.from_numpy(np.where((px_mask_numpy > 0) & np.isfinite(data) & (noise > 0),
                                                      1 / np.where(noise > 0, noise, 1.)**2, 0.)).to(DEVICE)
                            for data, noise in ((ref, ref_noise), (sci, sci_noise))]
ref = torch.from_numpy(np.nan_to_num(ref)).to(DEVICE)
sci = torch.from_numpy(np.nan_to_num(sci)).to(DEVICE)


def load_scored(path):
    """Cropped data, its noise, and inverse-variance weights on the pixel mask."""
    data, noise = load_and_crop_data(path, num_det_px=num_det_px, coron_type='MASK335R')
    good = (px_mask_numpy > 0) & np.isfinite(data) & (noise > 0)
    weights = np.where(good, 1 / np.where(good, noise, 1.)**2, 0.)
    return torch.from_numpy(np.where(good, data, 0.)).to(DEVICE), noise, torch.from_numpy(weights).to(DEVICE)


#### PENALTIES
supports = [diffnircam.optical_system.JWSTPrimary.amplitude.reshape(opds[0].shape) > 0,
            diffnircam.optical_system.NIRCamLyotstop.transmission().detach().reshape(opds[1].shape) > 1e-3]
post_stop = {}
diffnircam.optical_system.NIRCamLyotstop.register_forward_hook(lambda module, inputs, output: post_stop.update(field=output))


def window_share(PSF):
    """Detector window light over the light after the Lyot stop (any common normalisation)."""
    energy = (post_stop['field'].abs()**2 * wavefront_broad.wl_weights.view(-1, 1, 1)).sum()
    return PSF.sum() / energy


def dead_zone_penalty():
    """DEADLAM x mean over each map's support of relu(|map - mean| - t)^2 / t^2, summed over the maps."""
    total = 0.
    for grid, support in zip(opds, supports):
        m = grid[support] - grid[support].mean()
        total = total + (torch.relu(m.abs() - dead_zone)**2).mean() / dead_zone**2
    return dead_lam * total


def zero_map_share():
    """Window share of the current model with both maps at zero."""
    saved = [g.detach().clone() for g in opds]
    with torch.no_grad():
        for g in opds:
            g.zero_()
        PSF = diffnircam(wavefront_broad)
        share = window_share(PSF)
        for g, v in zip(opds, saved):
            g.copy_(v)
    return share


def save_state(path, data_file, data, weights):
    """Save the fitted state for data_file: maps, flat field, star, least-squares flux, geometry and options."""
    system = diffnircam.optical_system
    occulter, lyot_optic, detector_ = system.NIRCamcoronfocalplanemask, system.NIRCamLyotstop, system.NIRCamdetector
    with torch.no_grad():
        flux_correction = wavefront_broad.flux_correction.data.detach().clone()
        wavefront_broad.flux_correction.data.fill_(1.)
        raw = diffnircam(wavefront_broad)
        wavefront_broad.flux_correction.data.copy_(flux_correction)
        flux = ((weights * raw * data).sum() / (weights * raw**2).sum()).item()
    axial = getattr(occulter, 'axial_offset', None)
    options = dict(model_dir=data_dir, data_file=data_file, px_mask=px_mask_path, num_det_px=num_det_px, oversample=oversample,
                   case=case, fpm_raster=None if case == 'flip' else (os.path.join(geometry_dir, 'smooth_MASK335R_raster.npz')
                                                                      if case.startswith('smooth') else fpm_raster),
                   um_per_arcsec=None if case == 'flip' else um_per_arcsec[case.split('_')[0][-3:]],
                   band_limited=case.startswith('smooth') or '_bl' in case, square_cells=case.endswith('_sq'),
                   mount=case == 'flip', lyot_file=None if case == 'flip' else lyot_file, fpm_focal_length=fpm_focal_length)
    np.savez(path, pupil=opds[0].detach().cpu().numpy()[0], lyot=opds[1].detach().cpu().numpy()[0],
             flat=detector_.detector_flat_field.grid.detach().cpu().numpy()[0], star=wavefront_broad.angles_offset.detach().cpu().numpy(),
             flux=flux, registration=np.array([float(detector_.subpixel_shift.shift_x), float(detector_.subpixel_shift.shift_y)]),
             blur=float(torch.exp(detector_.charge_diffusion.log_sigma)) / oversample,
             lyot_shift=np.array([lyot_optic.pupil_shifts.x_shift.item(), lyot_optic.pupil_shifts.y_shift.item()]),
             clocking=lyot_optic.pupil_shifts.rotation.item(), fpm_dz=0. if axial is None else float(axial),
             options=json.dumps(options))


#### FIT (testing.py)
sci_loss_weight = 0.005
measure_SNR_every = 300
snrs_arr, track_iters = [], []
share_zero, loss_zero, share_history = {}, {}, []
for i in tqdm.tqdm(range(scheduler_iters + 1), mininterval=30.0):
    optimizer.zero_grad()
    PSF = diffnircam(wavefront_broad)
    share = window_share(PSF) if total_fraction else None    # before any flux scaling
    stage_weight = 1. if i < stage_cutoffs[2] else sci_loss_weight
    if loss_kind == 'chi2':
        data, weights = (ref, ref_weights) if i < stage_cutoffs[2] else (sci, sci_weights)
        PSF = PSF * ((weights * PSF * data).sum() / (weights * PSF**2).sum()).detach()
        loss = (weights * (PSF - data)**2).sum() / (weights > 0).sum() * stage_weight
    else:
        if i < stage_cutoffs[2]:
            loss = F.smooth_l1_loss(PSF, ref, reduction='none')
        else:
            loss = F.l1_loss(PSF, sci, reduction='none') * sci_loss_weight
        loss = (loss * px_mask).sum() / px_mask.sum()
    if total_fraction:
        frame = 'ref' if i < stage_cutoffs[2] else 'sci'
        if frame not in share_zero:
            share_zero[frame] = zero_map_share().detach()
            loss_zero[frame] = loss.detach() / stage_weight
        ratio = share / share_zero[frame]
        loss = loss + 35 * loss_zero[frame] * torch.relu(total_fraction - ratio)**2 * stage_weight
        if i % 100 == 0:
            share_history.append([i, ratio.item()])
    if dead_zone:
        loss = loss + dead_zone_penalty() * stage_weight
    loss.backward()
    if loss_kind != 'chi2' and i < stage_cutoffs[0]:
        wavefront_broad = hardcoded_match_flux(PSF, ref, px_mask, wavefront_broad)
    elif loss_kind != 'chi2' and i > stage_cutoffs[2] and i < stage_cutoffs[3]:
        wavefront_broad = hardcoded_match_flux(PSF, sci, px_mask, wavefront_broad)
    default_stages(diffnircam, wavefront_broad, i, stage_cutoffs=stage_cutoffs)
    optimizer.step()
    scheduler.step()
    if i == stage_cutoffs[2] - 1:
        calibrated = copy.deepcopy(diffnircam)                 # reference calibration, before the science stage
        calibrated_star = wavefront_broad.angles_offset.detach().clone()
        np.save(os.path.join(out_dir, 'residual_ref.npy'), (ref - PSF).detach().cpu().numpy())
        save_state(os.path.join(out_dir, 'fitted_reference.npz'), refdatapath, ref, ref_weights)
    if i % measure_SNR_every == 0 and i > stage_cutoffs[2]:
        snrs_arr.append(track_SNR((sci - PSF).detach().cpu().numpy(), i, fakes, injected_fluxes, mask_name='MASK335R',
                                  snr_smooth_sigma=1., px_mask_numpy=px_mask_numpy, save_plot=False)[2])
        track_iters.append(i)
    if i == stage_cutoffs[-1]:
        residual_sci = (sci - PSF).detach().cpu().numpy()
        np.save(os.path.join(out_dir, 'residual_sci.npy'), residual_sci)
        save_state(os.path.join(out_dir, 'fitted_final.npz'), sci_files[0], sci, sci_weights)
        break


#### SCORES
def chi2(PSF, frame):
    """Noise-weighted chi^2 / N with the flux solved in closed form, and that flux."""
    data, _, weights = frame
    flux = ((weights * PSF * data).sum() / (weights * PSF**2).sum()).detach()
    return ((weights * (flux * PSF - data)**2).sum() / (weights > 0).sum()).item(), flux


def refit_star(model, frame, star):
    """Star position from `star` by Adam with a cosine schedule, flux in closed form; chi^2 / N and the model image."""
    wavefront = copy.deepcopy(wavefront_broad)
    for parameter in wavefront.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        wavefront.angles_offset.copy_(star)
    wavefront.angles_offset.requires_grad_(True)
    star_optimizer = torch.optim.Adam([wavefront.angles_offset], lr=4e-9)               # [rad]
    star_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(star_optimizer, T_max=star_iters, eta_min=0.)
    for j in range(star_iters):
        star_optimizer.zero_grad()
        PSF = model(wavefront)
        data, _, weights = frame
        flux = ((weights * PSF * data).sum() / (weights * PSF**2).sum()).detach()
        ((weights * (flux * PSF - data)**2).sum()).backward()
        star_optimizer.step()
        star_scheduler.step()
    with torch.no_grad():
        PSF = model(wavefront)
        value, flux = chi2(PSF, frame)
    return value, (flux * PSF).cpu().numpy(), wavefront


for parameter in diffnircam.parameters():
    parameter.requires_grad_(False)
for parameter in calibrated.parameters():
    parameter.requires_grad_(False)
scores = {}
reference = load_scored(refdatapath)
with torch.no_grad():
    wavefront_ref = copy.deepcopy(wavefront_broad)
    wavefront_ref.angles_offset.copy_(calibrated_star)
    scores['reference chi2/N'] = chi2(calibrated(wavefront_ref), reference)[0]
held_chi2, injection_snr = [], {s: [] for s in injection_seps}
for path in held_files:
    frame = load_scored(path)
    value, model_image, wavefront = refit_star(calibrated, frame, calibrated_star)
    held_chi2.append(value)
    residual = np.squeeze(frame[0].cpu().numpy() - model_image)[None]
    for sep in injection_seps:
        for angle in injection_angles:
            planet = [{'pos_x_px': sep * np.cos(np.radians(angle)), 'pos_y_px': sep * np.sin(np.radians(angle))}]
            injected, flux = inject_fake_planets(np.zeros_like(residual), frame[1][None], wavefront, calibrated, planet, 5,
                                                 smooth_sigma_for_flux=1., DEVICE=DEVICE)
            injection_snr[sep].append(track_SNR(residual + injected, 0, planet, flux, mask_name='MASK335R', snr_smooth_sigma=1.,
                                                px_mask_numpy=px_mask_numpy, save_plot=False)[2][0])
scores['held-out chi2/N'] = float(np.mean(held_chi2))
science = [load_scored(path) for path in sci_files]
with torch.no_grad():
    scores['science 1 chi2/N'] = chi2(diffnircam(wavefront_broad), science[0])[0]
for k, frame in enumerate(science[1:], start=2):
    scores['science %d chi2/N' % k] = refit_star(diffnircam, frame, wavefront_broad.angles_offset.detach())[0]
scores['companion SNR, science 1'] = track_SNR(residual_sci, 0, [{'pos_x_px': companion[0], 'pos_y_px': companion[1]}], [1.],
                                               mask_name='MASK335R', snr_smooth_sigma=1., px_mask_numpy=px_mask_numpy, save_plot=False)[2][0]
scores['injected SNR, science 1 (testing.py)'] = track_SNR(residual_sci, 0, fakes, injected_fluxes, mask_name='MASK335R', snr_smooth_sigma=1.,
                                                          px_mask_numpy=px_mask_numpy, save_plot=False)[2][0]
for sep in injection_seps:
    scores['injected SNR, held-out, %g px' % sep] = float(np.mean(injection_snr[sep]))
scores['injected SNR, held-out, mean'] = float(np.mean([np.mean(v) for v in injection_snr.values()]))
grids = {'pupil': opds[0], 'lyot': opds[1]}
for name, grid in grids.items():
    support = (diffnircam.optical_system.JWSTPrimary.amplitude if name == 'pupil' else diffnircam.optical_system.NIRCamLyotstop.amplitude).reshape(grid.shape) > 0
    scores['%s map [nm]' % name] = 1e9 * grid.detach()[support].std().item()

if total_fraction:
    scores['window share / zero maps, end'] = share_history[-1][1]
np.savez(os.path.join(out_dir, 'scores.npz'), keys=np.array(list(scores)), values=np.array(list(scores.values())),
         share_history=np.array(share_history),
         held_chi2=np.array(held_chi2), snr_track=np.array(snrs_arr), snr_iters=np.array(track_iters),
         pupil=opds[0].detach().cpu().numpy()[0], lyot=opds[1].detach().cpu().numpy()[0])
print('case:', case)
for key, value in scores.items():
    print('%-40s %.4g' % (key, value))
