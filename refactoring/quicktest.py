import os 
os.environ["STPSF_PATH"] = "/projects/b1094/rodrigoyeah/optics_jwst/data/stpsf-data"
os.environ["PYSYN_CDBS"] = "/projects/b1094/rodrigoyeah/optics_jwst/data/synphot-data"

from model_building import create_STPSF_model, generate_STPSF_model_params, assemble_JWST_NIRCam_coron
from wavefronts import BroadbandWavefront
from model_building import assemble_JWST_NIRCam_coron
from optimization_handlers import default_stages
import torch.nn as nn
import torch 
from dl_utils import load_model_data, load_and_crop_data, inject_fake_planets, hardcoded_match_flux
import numpy as np
import tqdm
import torch.nn.functional as F



# CREATE OPTICAL SYSTEM

stpsfmodel = create_STPSF_model()

generate_STPSF_model_params(stpsfmodel,folder_to_save='/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/uuu')

diffnircam = assemble_JWST_NIRCam_coron('/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/uuu',80)


### CREATE WAVEFRONT
contrast_normalization = 0.003716380479003919
sim_to_real_scaling = 2.1722325193439986 # from comparing the simulation with the peak brightness of the real data
photons_normalization = 90578.00102527262 * sim_to_real_scaling *1e4 / 2251.24456327477 # mJy/sr
peak_flux_star  = nn.Parameter(torch.FloatTensor([photons_normalization / contrast_normalization]))

wf_npix = 1024
diameter = 6.603464
DEVICE = 'cuda'
num_wl = 21
oversample = 2
data_dir = '/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/HIP65426/model_setup'
num_det_px = 80

# Stellar position
offset_STAR = nn.Parameter(torch.FloatTensor([0., 0.])).to(DEVICE)

# TODO: wrapper function to do all of this at once for the simpler cases, including using STPSF to generate the masks and such
_, _, _, _, wlen_weights, _ = load_model_data(data_dir='/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/uuu')

# Set up the wavefront object: this will be propagated by our optical model
wavefront_broad = BroadbandWavefront(wf_npix, diameter, wlen_weights, peak_flux_star, offset_STAR).to(DEVICE)

result = diffnircam(wavefront_broad)

result = result.detach().cpu().numpy()

np.save('/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/resAUUU.npy', result)