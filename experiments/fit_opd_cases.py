"""OPD maps fitted on one reference frame for several focal plane masks, with the
instrument geometry held at the values fit_geometry_cases.py found for each mask, then
scored on held-out frames with only the star position and flux refitted.

Fitted on the training frame: the telescope (JWSTPrimary) and NIRCam (NIRCamLyotstop)
OPD maps and the star position. Held fixed: detector registration and blur, Lyot stop
shift and clocking, FPM axial offset. The loss is the noise-weighted chi^2 over the pixel
mask, with the flux solved in closed form at every iteration. The cases, the mask files
and the Lyot stop are those of fit_geometry_cases.py; GEOMETRY_DIR is its output.

Paths and options below, or from the environment variables named there. Prints chi^2/N
on the training and held-out frames and the map sizes per case, and saves the maps and
values to OUT_DIR.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'refactoring'))
import time
import numpy as np
import torch
from wavefronts import BroadbandWavefront
from model_building import assemble_JWST_NIRCam_coron
from dl_utils import load_model_data, load_and_crop_data, rad2arcsec

#### OPTIONS
data_dir = os.environ.get('MODEL_DIR', '/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/uuu')
train_file = os.environ['TRAIN_FILE']                        # reference frame the maps are fitted on
held_files = os.environ['HELD_FILES'].split(',')             # reference frames scored with the maps fixed
px_mask_path = os.environ.get('PX_MASK', '/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/HIP65426/model_setup/pixel_masks/pixel_masks.npy')
geometry_dir = os.environ['GEOMETRY_DIR']                    # output of fit_geometry_cases.py
fpm_raster = os.environ.get('FPM_RASTER')                    # real mask raster (.npz), needed for the real cases
lyot_file = os.environ['LYOT_FILE']                          # stpsf's MASKRND transmission, 1024 x 1024
out_dir = os.environ.get('OUT_DIR', 'opd_cases')
cases = os.environ.get('CASES', 'smooth593,smooth637,real593_bl,real637_bl').split(',')
iters = int(os.environ.get('ITERS', 1500))
held_iters = int(os.environ.get('HELD_ITERS', 200))
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


train = load_frame(train_file)
held = [load_frame(path) for path in held_files]
_, _, _, _, wlen_weights, _ = load_model_data(data_dir=data_dir)
lyot_amplitude = torch.FloatTensor(np.load(lyot_file)).to(DEVICE)[None]
geometry_file = np.load(os.path.join(geometry_dir, 'geometry_cases.npz'))
fitted = {case: dict(zip(geometry_file['keys'], values)) for case, values in zip(geometry_file['cases'], geometry_file['values'])}


#### MODEL AND FIT
def build(case):
    """Wavefront and instrument for one case at its fitted geometry, nothing free yet."""
    mask, scale = case.split('_')[0][:-3], case.split('_')[0][-3:]
    geometry = fitted[case]
    wavefront = BroadbandWavefront(wf_npix, diameter, wlen_weights, torch.ones(1, dtype=torch.float64),
                                   angles_offset=torch.zeros(2, dtype=torch.float64), DEVICE=DEVICE).to(DEVICE)
    instrument = assemble_JWST_NIRCam_coron(data_dir, num_det_px, DEVICE=DEVICE, oversample=oversample, babinet=True, mount=False,
                                            fpm_raster=os.path.join(geometry_dir, 'smooth_MASK335R_raster.npz') if mask == 'smooth' else fpm_raster,
                                            um_per_arcsec=um_per_arcsec[scale], band_limited=mask == 'smooth' or case.endswith('_bl'),
                                            fpm_axial_offset=1e-3 * geometry['FPM axial offset [mm]'], fpm_focal_length=fpm_focal_length)
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
    """Noise-weighted chi^2 / N with the flux solved in closed form, and that flux."""
    ref, weights, n_px = frame
    flux = ((weights * PSF * ref).sum() / (weights * PSF**2).sum()).detach()
    return (weights * (flux * PSF - ref)**2).sum() / n_px, flux


def fit(wavefront, instrument, frame, groups, n_iters):
    """Adam with a cosine schedule on the given (parameters, learning rate) groups; final chi^2/N."""
    for parameters, _ in groups:
        for parameter in parameters:
            parameter.requires_grad_(True)
    optimizer = torch.optim.Adam([{'params': p, 'lr': lr} for p, lr in groups])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=0.)
    for i in range(n_iters):
        optimizer.zero_grad()
        loss, _ = chi2(instrument(wavefront), frame)
        loss.backward()
        optimizer.step()
        scheduler.step()
    for parameters, _ in groups:
        for parameter in parameters:
            parameter.requires_grad_(False)
    with torch.no_grad():
        return chi2(instrument(wavefront), frame)[0].item()


def rms_nm(grid, amplitude):
    """RMS of a map over the support of a transmission [nm]."""
    support = amplitude.reshape(grid.shape) > 0
    return 1e9 * grid[support].std().item()


results = {}
for case in cases:
    t0 = time.time()
    wavefront, instrument = build(case)
    pupil_map = instrument.optical_system.JWSTPrimary.wfe_offsets.grid
    lyot_map = instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid
    with torch.no_grad():
        zero_maps = chi2(instrument(wavefront), train)[0].item()
    star_only = fit(wavefront, instrument, train, [([wavefront.angles_offset], 4e-9)], held_iters)
    trained = fit(wavefront, instrument, train, [([pupil_map, lyot_map], 4e-9), ([wavefront.angles_offset], 4e-9)], iters)
    star = wavefront.angles_offset.detach().clone()
    held_chi2 = []
    for frame in held:
        with torch.no_grad():
            wavefront.angles_offset.copy_(star)
        held_chi2.append(fit(wavefront, instrument, frame, [([wavefront.angles_offset], 4e-9)], held_iters))
    results[case] = {'train chi2/N, maps 0': star_only, 'train chi2/N': trained, 'held-out chi2/N': np.mean(held_chi2),
                     'pupil map [nm]': rms_nm(pupil_map, instrument.optical_system.JWSTPrimary.amplitude),
                     'Lyot map [nm]': rms_nm(lyot_map, lyot_amplitude),
                     'star x [mas]': 1e3 * rad2arcsec(star[0].item()), 'star y [mas]': 1e3 * rad2arcsec(star[1].item())}
    np.savez(os.path.join(out_dir, 'maps_%s.npz' % case), pupil=pupil_map.detach().cpu().numpy()[0],
             lyot=lyot_map.detach().cpu().numpy()[0], held_chi2=np.array(held_chi2))
    print(case, '| train chi2/N: geometry only %.3f, star refitted %.3f, with maps %.3f | held-out %s | %.0f s'
          % (zero_maps, star_only, trained, ' '.join('%.2f' % c for c in held_chi2), time.time() - t0), flush=True)
    del wavefront, instrument
    torch.cuda.empty_cache()

#### REPORT
keys = list(next(iter(results.values())))
print('\ncase', *keys, sep=' | ')
for case, values in results.items():
    print(case, *['%.4g' % values[k] for k in keys], sep=' | ')
np.savez(os.path.join(out_dir, 'opd_cases.npz'), cases=np.array(list(results)), keys=np.array(keys),
         values=np.array([[results[c][k] for k in keys] for c in results]))
