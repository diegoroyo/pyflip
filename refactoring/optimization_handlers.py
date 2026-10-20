import numpy as np 
import torch 
from optical_elements import DiffLyotOptic

def geometry_parameters(instrument):
    """Fitted instrument geometry: mask axial offsets, Lyot stop shift and rotation."""
    parameters = []
    for element in instrument.optical_system.values():
        if isinstance(getattr(element, 'axial_offset', None), torch.nn.Parameter):
            parameters.append(element.axial_offset)
        if isinstance(element, DiffLyotOptic):
            parameters += [element.pupil_shifts.x_shift, element.pupil_shifts.y_shift, element.pupil_shifts.rotation]
    return parameters


def scale_geometry_grads(instrument, factor):
    """Scale the geometry gradients in place."""
    for parameter in geometry_parameters(instrument):
        if factor == 0:
            parameter.grad = None   # a zero gradient still moves Adam through its moments
        elif parameter.grad is not None:
            parameter.grad *= factor


# basic handler assuming the simplest scene, default instrument parameters, and default parameter names
def default_stages(instrument, wavefront, stage_number, stage_cutoffs = [600, 3500, 5500, 6100, 11000]):
    # Zero-out gradients of parameters we are not optimizing per stage
    # Technically we don't need to multiply by 1, but keep for now for clarity and see if it slows down the code too much
    if stage_number < stage_cutoffs[0]:
        # Only fit star position
        wavefront.angles_offset.grad*=1.
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 0.
        instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad*= 0.
        scale_geometry_grads(instrument, 0.0)
    
    elif stage_number > stage_cutoffs[0] and stage_number < stage_cutoffs[1]:
        # Stop fitting position and flux; only fit OPDs
        wavefront.angles_offset.grad*=0.
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad*= 0.
        scale_geometry_grads(instrument, 1.0)

    elif stage_number > stage_cutoffs[1] and stage_number < stage_cutoffs[2]:
        # Fit OPDs and detector flats
        wavefront.angles_offset.grad*=0
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad*= 1.
        scale_geometry_grads(instrument, 1.0)
    
    elif stage_number > stage_cutoffs[2] and stage_number < stage_cutoffs[3]:
        # Now we should be on the science star, so again only fit star position
        wavefront.angles_offset.grad*=1.
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 0.
        instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad*= 0.
        scale_geometry_grads(instrument, 0.0)

    elif stage_number > stage_cutoffs[3]:
        # Fitting only one OPD in science star, and stop fitting position
        wavefront.angles_offset.grad*=0.
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad*= 0.
        scale_geometry_grads(instrument, 0.0)    