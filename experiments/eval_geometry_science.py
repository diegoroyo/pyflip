"""Science frames scored with the instrument geometry held fixed: either PyFLIP as
shipped or the values fit_geometry_cases.py fitted on the reference star. Only the star position
and flux are fitted per frame; no OPD maps. The loss is the noise-weighted chi^2 over the
pixel mask, with the flux solved in closed form at every iteration; nothing is masked
around companions, so a companion adds the same residual to every geometry.

PyFLIP as shipped: its mask file (FFT occulter, with the mount), the model directory's
Lyot stop as stored, no FPM axial offset, registration (-0.12, -0.11) pixels, blur 0.28
pixels. The masks are those of fit_geometry_cases.py; '_sq' appended treats the
band-limited raster's samples as square cells.

Frames as in fit_geometry_cases.py (FRAMES = 'model_dir:data_file,...'). Paths and
options below, or from the environment variables named there. Prints chi^2/N per
geometry and frame and saves them, the data and the model images to OUT_DIR/science_chi2.npz.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'refactoring'))
import time
import numpy as np
import torch
from wavefronts import BroadbandWavefront
from model_building import assemble_JWST_NIRCam_coron
from dl_utils import load_model_data, load_and_crop_data

#### OPTIONS
frames = [tuple(f.split(':')) for f in os.environ['FRAMES'].split(',')]
px_mask_path = os.environ.get('PX_MASK', '/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/HIP65426/model_setup/pixel_masks/pixel_masks.npy')
geometry_dir = os.environ['GEOMETRY_DIR']                    # output of fit_geometry_cases.py
fpm_raster = os.environ.get('FPM_RASTER')                    # real mask raster (.npz), needed for the real cases
lyot_file = os.environ['LYOT_FILE']                          # stpsf's MASKRND transmission, 1024 x 1024
out_dir = os.environ.get('OUT_DIR', 'science_chi2')
cases = os.environ.get('CASES', 'real593_bl_sq,real637_bl_sq').split(',')
iters = int(os.environ.get('ITERS', 300))
fpm_focal_length = 131.4                                     # equivalent focal length [m]
um_per_arcsec = {'593': 593.336, '637': 637.045}

DEVICE = os.environ.get('DEVICE', 'cuda')
wf_npix = 1024
diameter = 6.603464
num_det_px = 80
oversample = 2
os.makedirs(out_dir, exist_ok=True)

#### DATA
px_mask_file = np.load(px_mask_path).astype(bool)


def load_frame(path):
    """Cropped data, inverse-variance weights on the pixel mask, and the number of pixels used."""
    ref, ref_noise = load_and_crop_data(path, num_det_px=num_det_px, coron_type='MASK335R')
    px_mask = px_mask_file & np.isfinite(ref) & (ref_noise > 0)
    return (torch.from_numpy(np.where(px_mask, ref, 0.)).to(DEVICE),
            torch.from_numpy(np.where(px_mask, 1 / np.where(px_mask, ref_noise, 1.)**2, 0.)).to(DEVICE), int(px_mask.sum()))


data = [load_frame(path) for _, path in frames]
_, _, _, _, wlen_weights, _ = load_model_data(data_dir=frames[0][0])
lyot_amplitude = torch.FloatTensor(np.load(lyot_file)).to(DEVICE)[None]
geometry_file = np.load(os.path.join(geometry_dir, 'geometry_cases.npz'))
fitted = {case: dict(zip(geometry_file['keys'], values)) for case, values in zip(geometry_file['cases'], geometry_file['values'])}
geometries = {'PyFLIP as shipped': ('flip', None)}
geometries.update({'fitted, ' + case: (case, fitted[case.replace('_sq', '')]) for case in cases if case.replace('_sq', '') in fitted})


#### MODEL AND FIT
def build(model_dir, case, geometry):
    """Wavefront and instrument for one frame at a fixed geometry (PyFLIP's for case 'flip'), nothing free yet."""
    wavefront = BroadbandWavefront(wf_npix, diameter, wlen_weights, torch.ones(1, dtype=torch.float64),
                                   angles_offset=torch.zeros(2, dtype=torch.float64), DEVICE=DEVICE).to(DEVICE)
    if case == 'flip':
        instrument = assemble_JWST_NIRCam_coron(model_dir, num_det_px, DEVICE=DEVICE, oversample=oversample)
        for parameter in list(wavefront.parameters()) + list(instrument.parameters()):
            parameter.requires_grad_(False)
        return wavefront, instrument
    mask, scale = case.split('_')[0][:-3], case.split('_')[0][-3:]
    instrument = assemble_JWST_NIRCam_coron(model_dir, num_det_px, DEVICE=DEVICE, oversample=oversample, babinet=True, mount=False,
                                            fpm_raster=os.path.join(geometry_dir, 'smooth_MASK335R_raster.npz') if mask == 'smooth' else fpm_raster,
                                            um_per_arcsec=um_per_arcsec[scale], band_limited=mask == 'smooth' or '_bl' in case,
                                            square_cells=case.endswith('_sq'), fpm_axial_offset=1e-3 * geometry['FPM axial offset [mm]'],
                                            fpm_focal_length=fpm_focal_length)
    for parameter in list(wavefront.parameters()) + list(instrument.parameters()):
        parameter.requires_grad_(False)
    lyot = instrument.optical_system.NIRCamLyotstop
    lyot.amplitude = lyot_amplitude
    detector = instrument.optical_system.NIRCamdetector
    detector.subpixel_shift.shift_x = float(geometry['registration x [px]'])
    detector.subpixel_shift.shift_y = float(geometry['registration y [px]'])
    with torch.no_grad():
        lyot.pupil_shifts.x_shift.fill_(wf_npix * geometry['Lyot shift x [% D]'] / 100)
        lyot.pupil_shifts.y_shift.fill_(wf_npix * geometry['Lyot shift y [% D]'] / 100)
        lyot.pupil_shifts.rotation.fill_(geometry['Lyot clocking [deg]'])
        detector.charge_diffusion.log_sigma.fill_(np.log(geometry['blur [px]'] * oversample))
    return wavefront, instrument


def chi2(PSF, frame):
    """Noise-weighted chi^2 / N with the flux solved in closed form."""
    ref, weights, n_px = frame
    flux = ((weights * PSF * ref).sum() / (weights * PSF**2).sum()).detach()
    return (weights * (flux * PSF - ref)**2).sum() / n_px


def fit_star(wavefront, instrument, frame):
    """Star position by Adam with a cosine schedule; final chi^2/N and the model image at the fitted flux."""
    wavefront.angles_offset.requires_grad_(True)
    optimizer = torch.optim.Adam([wavefront.angles_offset], lr=4e-9)                     # [rad]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)
    for i in range(iters):
        optimizer.zero_grad()
        chi2(instrument(wavefront), frame).backward()
        optimizer.step()
        scheduler.step()
    with torch.no_grad():
        PSF = instrument(wavefront)
        ref, weights, _ = frame
        flux = (weights * PSF * ref).sum() / (weights * PSF**2).sum()
        return chi2(PSF, frame).item(), np.squeeze((flux * PSF).cpu().numpy())


results, models = {}, {}
for name, (case, geometry) in geometries.items():
    t0 = time.time()
    results[name], models[name] = [], []
    for (model_dir, _), frame in zip(frames, data):
        wavefront, instrument = build(model_dir, case, geometry)
        value, image = fit_star(wavefront, instrument, frame)
        results[name].append(value)
        models[name].append(image)
        del wavefront, instrument
        torch.cuda.empty_cache()
    print(name, '| chi2/N per frame', ' '.join('%.2f' % c for c in results[name]), '| mean %.2f | %.0f s'
          % (np.mean(results[name]), time.time() - t0), flush=True)

#### REPORT
for k, (model_dir, path) in enumerate(frames):
    print('frame %d: %s, %s' % (k, model_dir, os.path.basename(path)))
np.savez(os.path.join(out_dir, 'science_chi2.npz'), geometries=np.array(list(results)), chi2=np.array(list(results.values())),
         frames=np.array([':'.join(f) for f in frames]), models=np.array(list(models.values())),
         data=np.array([np.squeeze(frame[0].cpu().numpy()) for frame in data]),
         weights=np.array([np.squeeze(frame[1].cpu().numpy()) for frame in data]))
