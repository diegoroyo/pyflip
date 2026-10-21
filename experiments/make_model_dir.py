"""A PyFLIP model directory made with create_STPSF_model and generate_STPSF_model_params
for one NIRCam filter with MASK335R and the round Lyot stop, at the telescope OPD
closest to OBS_TIME. The source is a blackbody of temperature T_STAR (STPSF's
synthetic spectra need the Phoenix grid). Needs STPSF, its data and PYSYN_CDBS.

Paths and options below, or from the environment variables named there.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'refactoring'))
import astropy.units as u
from synphot import SourceSpectrum
from synphot.models import BlackBodyNorm1D
from model_building import create_STPSF_model, generate_STPSF_model_params

#### OPTIONS
filter_name = os.environ.get('FILTER', 'F410M')
num_wl = int(os.environ.get('NUM_WL', 9))                    # 9 for medium filters, 21 for wide ones
obs_time = os.environ.get('OBS_TIME', '2022-07-29T23:00:00')
t_star = float(os.environ.get('T_STAR', 9500.))              # [K]
folder = os.environ.get('OUT_DIR', 'model_%s_n%d' % (filter_name, num_wl))

instrument = create_STPSF_model(inst_params={'filter': filter_name, 'image_mask': 'MASK335R', 'pupil_mask': 'CIRCLYOT'},
                                obs_time=obs_time)
spectrum = SourceSpectrum(BlackBodyNorm1D, temperature=t_star * u.K)
generate_STPSF_model_params(instrument, input_spectrum=spectrum, num_wl=num_wl, folder_to_save=folder)
print('model directory:', folder)
