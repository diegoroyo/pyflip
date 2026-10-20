"""Instrument geometry fitted on one or more reference frames for several focal plane
masks, with no OPD maps. Shared by all frames: detector registration and blur, Lyot stop
shift and clocking, and FPM axial offset. Per frame: star position and flux. The loss is
the noise-weighted chi^2 over the pixel mask, summed over the frames, with each flux
solved in closed form at every iteration. One filter: the one of the model directories.

Each frame has its own model directory (for its own telescope OPD) and data file, given
as FRAMES = 'model_dir:data_file,model_dir:data_file,...'; without FRAMES, the frame is
MODEL_DIR:REF_FILE.

Cases: the smooth MASK335R profile and the real mask raster (every cell, and
band-limited; '_sq' appended treats its samples as square cells), each with the mask at
593.336 and 637.045 um/arcsec on the sky. The
smooth profile is drawn as a raster at its design scale (593.336 um/arcsec), so all
cases go through the same raster occulter. The mount is dropped. The Lyot stop is
LYOT_FILE (stpsf's MASKRND as stored), moved by the fitted shift and clocking.

Paths and options below, or from the environment variables named there. Prints the
fitted values and chi^2/N per case and frame and saves them to OUT_DIR/geometry_cases.npz.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'refactoring'))
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.special import j1, jn_zeros
from wavefronts import BroadbandWavefront
from model_building import assemble_JWST_NIRCam_coron
from dl_utils import load_model_data, load_and_crop_data, rad2arcsec

#### OPTIONS
data_dir = os.environ.get('MODEL_DIR', '/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/uuu')
refdatapath = os.environ.get('REF_FILE', '/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/HIP65426_newmodel/NEWGOODDATA/onlycalints_nanreplaced/jw01386001001_0310e_00001_nrcalong_calints.fits')
frames = [tuple(f.split(':')) for f in os.environ.get('FRAMES', data_dir + ':' + refdatapath).split(',')]
px_mask_path = os.environ.get('PX_MASK', '/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/HIP65426/model_setup/pixel_masks/pixel_masks.npy')
fpm_raster = os.environ.get('FPM_RASTER')                    # real mask raster (.npz), needed for the real cases
lyot_file = os.environ['LYOT_FILE']                          # stpsf's MASKRND transmission, 1024 x 1024
out_dir = os.environ.get('OUT_DIR', 'geometry_cases')
cases = os.environ.get('CASES', 'smooth593,smooth637,real593,real637,real593_bl,real637_bl').split(',')
iters = int(os.environ.get('ITERS', 600))
fpm_focal_length = 131.4                                     # equivalent focal length [m]
lyot_start = (-10.35, -7.84, -0.484)                         # stop shift x, y [samples] and clocking [deg] from LYOT_FILE
fpm_axial_start = -2.195e-3                                  # FPM axial offset [m]
blur_start = 0.35                                            # blur sigma [detector pixels]
um_per_arcsec = {'593': 593.336, '637': 637.045}

DEVICE = os.environ.get('DEVICE', 'cuda')
wf_npix = 1024
diameter = 6.603464
num_det_px = 80
oversample = 2
os.makedirs(out_dir, exist_ok=True)

#### SMOOTH PROFILE AS A RASTER
MASK335R_SIGMA = 3.2927866                                   # stpsf's MASK335R profile parameter [1/arcsec]


def smooth_raster(path, pitch_um=2., radius_um=1300., design_um_per_arcsec=593.336):
    """Write stpsf's MASK335R profile as a raster file of opaque amplitude.

    Parameters
    ----------
    path : str
        Output .npz, in the format of load_raster_region.
    pitch_um : float, optional
        Raster cell size [um].
    radius_um : float, optional
        Half width of the raster [um]; the profile is clear beyond 2.13 arcsec.
    design_um_per_arcsec : float, optional
        Scale at which the profile is drawn [um/arcsec].
    """
    x = (np.arange(2 * int(radius_um / pitch_um) + 1) - int(radius_um / pitch_um)) * pitch_um
    s = np.clip(MASK335R_SIGMA * np.hypot(*np.meshgrid(x, x)) / design_um_per_arcsec, np.finfo(float).tiny, jn_zeros(1, 2)[1])
    np.savez(path, opaque=(2 * j1(s) / s)**2, pitch_um=pitch_um, x0_um=x[0] - pitch_um / 2, y0_um=x[0] - pitch_um / 2,
             centroid_um=np.zeros(2))


smooth_path = os.path.join(out_dir, 'smooth_MASK335R_raster.npz')
smooth_raster(smooth_path)

#### DATA
px_mask_file = np.load(px_mask_path).astype(bool)
refs, weights, n_px = [], [], []
for _, path in frames:
    ref, ref_noise = load_and_crop_data(path, num_det_px=num_det_px, coron_type='MASK335R')
    px_mask = px_mask_file & np.isfinite(ref) & (ref_noise > 0)
    refs.append(torch.from_numpy(np.where(px_mask, ref, 0.)).to(DEVICE))
    weights.append(torch.from_numpy(np.where(px_mask, 1 / np.where(px_mask, ref_noise, 1.)**2, 0.)).to(DEVICE))
    n_px.append(int(px_mask.sum()))
_, _, _, _, wlen_weights, _ = load_model_data(data_dir=frames[0][0])
lyot_amplitude = torch.FloatTensor(np.load(lyot_file)).to(DEVICE)[None]
shared_elements = ('NIRCamcoronfocalplanemask', 'NIRCamLyotstop', 'NIRCamdetector')


#### MODEL AND FIT
def build(case):
    """Wavefronts and instruments of all frames for one case, with the fitted parameters at
    their starting values; the occulter, Lyot stop and detector are shared by the frames."""
    mask, scale = case.split('_')[0][:-3], case.split('_')[0][-3:]
    wavefronts, instruments = [], []
    for model_dir, _ in frames:
        wavefront = BroadbandWavefront(wf_npix, diameter, wlen_weights, torch.ones(1, dtype=torch.float64),
                                       angles_offset=torch.zeros(2, dtype=torch.float64), DEVICE=DEVICE).to(DEVICE)
        if instruments:
            instrument = assemble_JWST_NIRCam_coron(model_dir, num_det_px, DEVICE=DEVICE, oversample=oversample)
            for name in shared_elements:
                instrument.optical_system[name] = instruments[0].optical_system[name]
        else:
            instrument = assemble_JWST_NIRCam_coron(model_dir, num_det_px, DEVICE=DEVICE, oversample=oversample, babinet=True,
                                                    mount=False, fpm_raster=smooth_path if mask == 'smooth' else fpm_raster,
                                                    um_per_arcsec=um_per_arcsec[scale],
                                                    band_limited=mask == 'smooth' or '_bl' in case, square_cells=case.endswith('_sq'),
                                                    fpm_axial_offset=fpm_axial_start, fpm_focal_length=fpm_focal_length)
        for parameter in list(wavefront.parameters()) + list(instrument.parameters()):
            parameter.requires_grad_(False)
        wavefronts.append(wavefront)
        instruments.append(instrument)
    lyot = instruments[0].optical_system.NIRCamLyotstop
    lyot.amplitude = lyot_amplitude
    detector = instruments[0].optical_system.NIRCamdetector
    detector.subpixel_shift.shift_x = nn.Parameter(torch.zeros((), dtype=torch.float64, device=DEVICE))
    detector.subpixel_shift.shift_y = nn.Parameter(torch.zeros((), dtype=torch.float64, device=DEVICE))
    with torch.no_grad():
        lyot.pupil_shifts.x_shift.fill_(lyot_start[0])
        lyot.pupil_shifts.y_shift.fill_(lyot_start[1])
        lyot.pupil_shifts.rotation.fill_(lyot_start[2])
        detector.charge_diffusion.log_sigma.fill_(np.log(blur_start * oversample))
    groups = {'star position': ([wavefront.angles_offset for wavefront in wavefronts], 4e-9),    # [rad]
              'registration': ([detector.subpixel_shift.shift_x, detector.subpixel_shift.shift_y], 5e-3),   # [pixels]
              'blur': ([detector.charge_diffusion.log_sigma], 5e-3),
              'Lyot stop shift': ([lyot.pupil_shifts.x_shift, lyot.pupil_shifts.y_shift], 2e-2),          # [samples]
              'Lyot stop clocking': ([lyot.pupil_shifts.rotation], 2e-3),                                 # [deg]
              'FPM axial offset': ([instruments[0].optical_system.NIRCamcoronfocalplanemask.axial_offset], 2e-5)}  # [m]
    for parameters, _ in groups.values():
        for parameter in parameters:
            parameter.requires_grad_(True)
    return wavefronts, instruments, groups


def chi2(PSF, k):
    """Noise-weighted chi^2 / N of frame k with the flux solved in closed form, and that flux."""
    flux = ((weights[k] * PSF * refs[k]).sum() / (weights[k] * PSF**2).sum()).detach()
    return (weights[k] * (flux * PSF - refs[k])**2).sum() / n_px[k], flux


def geometry(wavefronts, instrument):
    """Fitted values in reporting units."""
    detector = instrument.optical_system.NIRCamdetector
    shifts = instrument.optical_system.NIRCamLyotstop.pupil_shifts
    values = {'registration x [px]': detector.subpixel_shift.shift_x.item(),
              'registration y [px]': detector.subpixel_shift.shift_y.item(),
              'blur [px]': np.exp(detector.charge_diffusion.log_sigma.item()) / oversample,
              'Lyot shift x [% D]': 100 * shifts.x_shift.item() / wf_npix,
              'Lyot shift y [% D]': 100 * shifts.y_shift.item() / wf_npix,
              'Lyot clocking [deg]': shifts.rotation.item(),
              'FPM axial offset [mm]': 1e3 * instrument.optical_system.NIRCamcoronfocalplanemask.axial_offset.item()}
    for k, wavefront in enumerate(wavefronts):
        values['star x %d [mas]' % k] = 1e3 * rad2arcsec(wavefront.angles_offset[0].item())
        values['star y %d [mas]' % k] = 1e3 * rad2arcsec(wavefront.angles_offset[1].item())
    return values


results = {}
for case in cases:
    t0 = time.time()
    wavefronts, instruments, groups = build(case)
    optimizer = torch.optim.Adam([{'params': p, 'lr': lr} for p, lr in groups.values()])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)
    history = []
    for i in range(iters):
        optimizer.zero_grad()
        total = 0.
        for k in range(len(frames)):
            loss, _ = chi2(instruments[k](wavefronts[k]), k)
            (loss * n_px[k] / sum(n_px)).backward()
            total += loss.item() * n_px[k] / sum(n_px)
        optimizer.step()
        scheduler.step()
        history.append(total)
    values = geometry(wavefronts, instruments[0])
    with torch.no_grad():
        for k in range(len(frames)):
            final, flux = chi2(instruments[k](wavefronts[k]), k)
            values['chi2/N %d' % k] = final.item()
            values['flux %d' % k] = flux.item()
    values['chi2/N all'] = sum(values['chi2/N %d' % k] * n_px[k] for k in range(len(frames))) / sum(n_px)
    results[case] = values
    np.save(os.path.join(out_dir, 'history_%s.npy' % case), np.array(history))
    print(case, '| chi2/N start %.3f, final %.3f, change over the last 10 %% of iterations %.1e | %.0f s'
          % (history[0], values['chi2/N all'], history[-iters // 10] - history[-1], time.time() - t0), flush=True)
    del wavefronts, instruments, optimizer
    torch.cuda.empty_cache()

#### REPORT
for k, (model_dir, path) in enumerate(frames):
    print('frame %d: %s, %s' % (k, model_dir, os.path.basename(path)))
keys = list(next(iter(results.values())))
print('\ncase', *keys, sep=' | ')
for case, values in results.items():
    print(case, *['%.4g' % values[k] for k in keys], sep=' | ')
np.savez(os.path.join(out_dir, 'geometry_cases.npz'), cases=np.array(list(results)), keys=np.array(keys),
         values=np.array([[results[c][k] for k in keys] for c in results]), frames=np.array([':'.join(f) for f in frames]))
