import torch.nn as nn
import torch
import numpy as np 
import sys
from scipy import ndimage
if __package__:
    from .model_classes import OPDOffsetModule, LearnableGaussianBlur, FlatFieldingModule, DetSubPixShift, ShiftModule
    from .dl_utils import crop_to, partial_MFT, shift_image_subpixel, centred_axis
    from .diffraction import FresnelTransfer, propagate_wavelengths
    from .bluestein_fft import bluestein_fft
else:
    from model_classes import OPDOffsetModule, LearnableGaussianBlur, FlatFieldingModule, DetSubPixShift, ShiftModule
    from dl_utils import crop_to, partial_MFT, shift_image_subpixel, centred_axis
    from diffraction import FresnelTransfer, propagate_wavelengths
    from bluestein_fft import bluestein_fft

class DiffOpticalElement(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.optic_type = None

    def forward(self):
        # The forward function is called by calling the class itself.
        # DiffOpticalElement(input) = DiffOpticalElement.forward(input)
        pass


class DiffPupilOptic(DiffOpticalElement):
    def __init__(self,name=None, opd=None, amplitude=None,wfe_offsets=None):
        super().__init__()
        self.name = name
        self.amplitude = amplitude
        self.opd = opd
        self.optic_type = 'pupil'
        if wfe_offsets is None: 
            self.wfe_offsets = OPDOffsetModule(opd.shape[-2], opd.shape[-1])
        else:
            self.wfe_offsets = wfe_offsets

        self.pupil_shifts = ShiftModule(amplitude.shape[-2], amplitude.shape[-1], device=amplitude.device)

    def forward(self,phasors, wls, normalize=False):
        # Phasors is the result of broadbandwavefront.get_phasors().
        # Ideally, that is run at the DiffOpticalSystem level, so these only need to
        # take in the phasors and do the math.
        # TODO: be more elegant about this, if it does not compromise speed.

        # Through OPD
        layer = self.wfe_offsets(self.opd)
        wfe = torch.exp(1j * layer.unsqueeze(0) * 2 * np.pi / wls.unsqueeze(1).unsqueeze(2))
        phasors = phasors * wfe
    
        # Through amplitude transmission
        new_phasor = phasors * self.transmission()
        if normalize:
            denom = new_phasor.abs() ** 2
            denom = torch.sum(denom, dim=(-2, -1), keepdim=True) ** 0.5
        else:
            denom = 1.
        return new_phasor / denom

    def transmission(self):
        """Amplitude transmission after pupil_shifts."""
        return self.pupil_shifts(self.amplitude)
    

class DiffLyotOptic(DiffPupilOptic):
    """Lyot stop with optional axial displacement from the nominal pupil plane.

    Attributes
    ----------
    axial_offset : torch.nn.Parameter or None
        Stop displacement [m] in the equivalent optical system used for pupil
        sampling. Positive values move downstream. None uses the nominal stop.
    pupil_shifts : ShiftModule
        Shift and rotation of the stop transmission.

    The OPD remains at the nominal pupil plane, after propagation back from the
    displaced stop. Only the amplitude transmission moves with the stop.
    """
    def __init__(self, name=None, opd=None, amplitude=None, wfe_offsets=None, axial_offset=None):
        super().__init__(name, opd, amplitude, wfe_offsets)
        self.axial_offset = None
        if axial_offset is not None:
            self.axial_offset = nn.Parameter(torch.as_tensor(
                axial_offset, dtype=torch.float64, device=amplitude.device))

    def forward(self, phasors, wls, normalize=False, d_pupil=None):
        """Apply the stop using Fresnel propagation to its displaced plane.

        Parameters
        ----------
        phasors : torch.Tensor
            Complex pupil field, shape (..., n_wl, n, n). Retain the full grid
            returned from the focal plane when axial displacement is enabled.
        wls : torch.Tensor
            Wavelengths of the light [m], shape (n_wl,).
        normalize : bool, optional
            Normalize the transmitted field to unit summed intensity.
        d_pupil : float or torch.Tensor, optional
            Spatial sampling interval of the pupil field [m]. Required for
            axial displacement.

        Returns
        -------
        torch.Tensor
            Field at the nominal pupil plane, cropped to the stop array size.
        """
        if self.axial_offset is None:
            return super().forward(phasors, wls, normalize)
        returned = self.displaced_stop(phasors, wls, d_pupil)
        layer = self.wfe_offsets(self.opd)
        wfe = torch.exp(1j * layer.unsqueeze(0) * 2 * np.pi / wls.unsqueeze(1).unsqueeze(2))
        new_phasor = returned * wfe
        if normalize:
            new_phasor = new_phasor / new_phasor.abs().square().sum(dim=(-2, -1), keepdim=True).sqrt()
        return new_phasor

    def displaced_stop(self, phasors, wls, d_pupil):
        """Apply the displaced stop transmission and return to the nominal pupil.

        Uses the same field and sampling arguments as forward, without applying
        an OPD. The output is cropped to the stop array size.
        """
        phasors = phasors.to(torch.complex128)
        if d_pupil is None:
            raise ValueError('axial Lyot displacement requires d_pupil')
        n_stop = self.amplitude.shape[-1]
        n_pad = phasors.shape[-1] - n_stop
        if n_pad < 0 or n_pad % 2:
            raise ValueError('pupil field must contain the stop on a grid of matching parity')
        transmission = torch.nn.functional.pad(self.transmission(), (n_pad // 2,) * 4)
        wavelengths = wls.to(device=phasors.device, dtype=torch.float64)
        propagators = [FresnelTransfer(d_pupil, wl) for wl in wavelengths]

        displaced = propagate_wavelengths(phasors, propagators, self.axial_offset)
        returned = propagate_wavelengths(displaced * transmission, propagators, -self.axial_offset)
        return crop_to(returned, n_stop)


class DiffImageOptic(DiffOpticalElement):
    def __init__(self,name=None, amplitude=None, axial_offset=None, focal_length=None):
        super().__init__()
        self.name =name
        self.amplitude = amplitude
        self.optic_type = 'image'
        self.focal_length = focal_length
        self.axial_offset = None
        if axial_offset is not None:
            if focal_length is None or focal_length <= 0:
                raise ValueError('axial displacement requires a positive equivalent focal length')
            self.axial_offset = nn.Parameter(torch.as_tensor(
                axial_offset, dtype=torch.float64, device=amplitude.device))

    def forward(self,phasor,npixels_in,oversample=2, wavelengths=None, d_pupil=None, crop_output=True):
        # Phasors is the result of broadbandwavefront.get_phasors().
        # Ideally, that is run at the DiffOpticalSystem level, so these only need to
        # take in the phasors and do the math.
        # TODO: be more elegant about this, if it does not compromise speed.

        # Propagate wavefront to image plane. TODO: only transform if needed
        if self.axial_offset is not None:
            phasor = phasor.to(torch.complex128)
        if oversample > 1:
            _npixels = (npixels_in * (oversample - 1)) // 2
            phasor = torch.nn.functional.pad(phasor, (_npixels, ) * 4)

        phasors = torch.fft.fftshift(torch.fft.ifft2(phasor), dim=[-2, -1])

        # Apply transmission (e.g., focal plaen mask)
        if self.axial_offset is None:
            new_phasor = phasors * self.amplitude
        else:
            new_phasor = self.displaced_mask(phasors, wavelengths, d_pupil)

        # Back to pupil plane and format. TODO: only transform if needed
        new_phasor = torch.fft.fft2(torch.fft.fftshift(new_phasor, dim=[-2, -1]))
        if crop_output:
            new_phasor = crop_to(new_phasor, npixels_in)

        return new_phasor

    def displaced_mask(self, phasors, wavelengths, d_pupil):
        """Apply an axially displaced focal plane mask using Fresnel diffraction.

        Parameters
        ----------
        phasors : torch.Tensor
            Complex field at the nominal focal plane, shape (..., n_wl, n, n).
            Requires an even number of samples across the full focal FFT grid.
        wavelengths : torch.Tensor
            Wavelengths of the light [m], shape (n_wl,).
        d_pupil : float or torch.Tensor
            Spatial sampling interval of the pupil field [m].

        Returns
        -------
        torch.Tensor
            Field after transmission through the displaced mask and Fresnel
            propagation back to the nominal focal plane.

        Notes
        -----
        Pupil sampling, focal length and axial displacement refer to the same
        equivalent optical system. Propagation uses the transfer function
        method, including at zero displacement.
        """
        phasors = phasors.to(torch.complex128)
        n = phasors.shape[-1]
        if n % 2:
            raise ValueError('axial masks currently require an even focal grid')
        if wavelengths is None or d_pupil is None:
            raise ValueError('axial masks require wavelengths and d_pupil')
        if phasors.shape[-3] != len(wavelengths):
            raise ValueError('the dimension before the spatial axes must index wavelength')

        # The unshifted pupil IFFT introduces a linear phase. Removing it gives the physical
        # focal field (with reversed axes), anti-periodic because the pupil samples
        # are half-integers. A half-bin frequency offset retains that boundary
        # condition without allocating a doubled focal window.
        axis = torch.arange(n, device=phasors.device, dtype=torch.float64) - n // 2
        phase_factor = torch.exp(2j * torch.pi * ((n - 1) / 2) / n *
                                 (axis[:, None] + axis[None, :])).to(phasors.dtype)
        wavelengths = wavelengths.to(device=phasors.device, dtype=torch.float64)
        focal_sampling = wavelengths * self.focal_length / (n * d_pupil)
        propagators = [
            FresnelTransfer(d_focal, wl, frequency_offset=0.5 / (n * d_focal))
            for wl, d_focal in zip(wavelengths, focal_sampling)
        ]

        displaced = propagate_wavelengths(phasors / phase_factor, propagators, self.axial_offset)
        returned = propagate_wavelengths(displaced * self.amplitude, propagators, -self.axial_offset)
        return returned * phase_factor
    

class BlockedRegion(nn.Module):
    """Part of a focal plane mask's blocked amplitude, 1 - t, on its own focal window.

    Attributes
    ----------
    blocked : torch.Tensor
        Blocked amplitude, shape (n_wl or 1, n_y, n_x).
    centre : tuple of float
        Window centre (y, x) relative to the optical axis [focal samples].
    n_fft : torch.Tensor
        Focal samples per period, wavelength / (d_pupil * d_focal), shape (n_wl,) or (1,).
    """
    def __init__(self, blocked, centre, n_fft):
        super().__init__()
        self.register_buffer('blocked', torch.as_tensor(np.asarray(blocked), dtype=torch.float64))
        self.register_buffer('n_fft', torch.as_tensor(np.asarray(n_fft), dtype=torch.float64).reshape(-1))
        self.centre = tuple(float(c) for c in centre)

    def propagate(self, field, n_out):
        """Field the region blocks, returned to the second pupil.

        Parameters
        ----------
        field : torch.Tensor
            Complex pupil field, shape (..., n_wl, n, n).
        n_out : int
            Number of output samples.

        Returns
        -------
        torch.Tensor
            Blocked field at the second pupil, shape (..., n_wl, n_out, n_out).
        """
        if len(self.n_fft) == 1:
            return self._propagate(field, self.blocked, float(self.n_fft), n_out)
        blocked = self.blocked.expand(len(self.n_fft), -1, -1)
        return torch.stack([self._propagate(field[..., i, :, :], blocked[i], float(n_fft), n_out)
                            for i, n_fft in enumerate(self.n_fft)], dim=-3)

    def _propagate(self, field, blocked, n_fft, n_out):
        focal = bluestein_fft(field, tuple(blocked.shape[-2:]), n_fft, centre_out=self.centre) / n_fft**2
        return bluestein_fft(focal * blocked.to(field.dtype), n_out, n_fft, inverse=True, centre_in=self.centre)


class BlockedBand(nn.Module):
    """Blocked amplitude uniform across whole focal rows; only y is transformed.

    Attributes
    ----------
    blocked : torch.Tensor
        Blocked amplitude per focal row, shape (n_wl or 1, M), optical axis at row M // 2.
    """
    def __init__(self, blocked):
        super().__init__()
        self.register_buffer('blocked', torch.as_tensor(np.asarray(blocked), dtype=torch.float64))

    def propagate(self, field, n_out):
        """Field the band blocks, returned to the second pupil, shape (..., n_wl, n_out, n_out)."""
        # The x transforms over one whole period cancel.
        n, M = field.shape[-1], self.blocked.shape[-1]
        padded = torch.nn.functional.pad(field, (0, 0, (M - n) // 2, (M - n) // 2))
        focal = torch.fft.fftshift(torch.fft.ifft(padded, dim=-2), dim=-2)
        returned = torch.fft.fft(torch.fft.fftshift(focal * self.blocked[..., :, None].to(field.dtype), dim=-2), dim=-2)
        returned = returned[..., (M - n_out) // 2:(M + n_out) // 2, :]
        return torch.nn.functional.pad(returned, ((n_out - n) // 2,) * 2)


class BabinetImageOptic(DiffOpticalElement):
    """Focal plane mask applied through Babinet's principle.

    Only the blocked amplitude b = 1 - t is propagated, on each region's window:
        E_p2 = E_p1 - sum over regions of T_back[ b * T[E_p1] ]

    Attributes
    ----------
    regions : nn.ModuleList of BlockedRegion or BlockedBand
        Disjoint parts of the blocked amplitude.
    axial_offset : torch.nn.Parameter or None
        Mask displacement along the beam [primary-equivalent m], positive downstream.
    focal_length : float
        Equivalent focal length [m].

    Notes
    -----
    The displacement is the pupil phase exp(-i pi dz r^2 / (wavelength F^2)) before
    the mask and its conjugate after: exact Fresnel propagation, as no focal field
    is truncated.
    """
    def __init__(self, regions, name=None, axial_offset=None, focal_length=None):
        super().__init__()
        self.name = name
        self.optic_type = 'image'
        self.regions = nn.ModuleList(regions)
        self.focal_length = focal_length
        self.axial_offset = None
        if axial_offset is not None:
            if focal_length is None or focal_length <= 0:
                raise ValueError('axial displacement requires a positive equivalent focal length')
            self.axial_offset = nn.Parameter(torch.as_tensor(axial_offset, dtype=torch.float64))

    def forward(self, phasor, npixels_in, oversample=2, wavelengths=None, d_pupil=None, crop_output=True):
        """Field at the second pupil, on npixels_in samples, or npixels_in * oversample if not cropped."""
        n_out = npixels_in if crop_output else npixels_in * oversample
        field = phasor
        if self.axial_offset is not None:
            field = field * self.defocus(npixels_in, wavelengths, d_pupil).to(field.dtype)
        blocked = sum(region.propagate(field, n_out) for region in self.regions)
        if self.axial_offset is not None:
            blocked = blocked * self.defocus(n_out, wavelengths, d_pupil).conj().to(field.dtype)
        return torch.nn.functional.pad(phasor, ((n_out - npixels_in) // 2,) * 4) - blocked

    def defocus(self, n, wavelengths, d_pupil):
        """Pupil phase of the mask displacement, shape (n_wl, n, n)."""
        axis = centred_axis(n, d_pupil, self.axial_offset.device)
        radius2 = axis[:, None].square() + axis[None, :].square()
        wavelengths = wavelengths.to(device=axis.device, dtype=torch.float64)[:, None, None]
        return torch.exp(-1j * torch.pi * self.axial_offset * radius2 / (wavelengths * self.focal_length**2))


def flip_mask_regions(fpm, spot=True, mount=True):
    """Split PyFLIP's focal plane mask files into a band and connected regions.

    Parameters
    ----------
    fpm : array_like
        Amplitude transmission on PyFLIP's focal grid, shape (n_wl, M, M), optical axis at M // 2.
    spot : bool, optional
        Keep the region on the optical axis.
    mount : bool, optional
        Keep the regions off the axis.

    Returns
    -------
    list of BlockedRegion and BlockedBand
    """
    blocked = 1 - np.asarray(fpm, dtype=np.float64).reshape((-1,) + np.shape(fpm)[-2:])
    M = blocked.shape[-1]
    axis = M // 2
    uniform = (blocked == blocked[..., :1]).all(-1) & (blocked[..., 0] != 0)
    band = np.where(uniform, blocked[..., 0], 0.)
    rest = blocked - band[..., None]
    regions = [BlockedBand(band)] if mount and band.any() else []
    labels, _ = ndimage.label(rest.any(0))
    for k, (ys, xs) in enumerate(ndimage.find_objects(labels), start=1):
        on_axis = ys.start <= axis < ys.stop and xs.start <= axis < xs.stop
        if spot if on_axis else mount:
            centre = ((ys.start + ys.stop - 1) / 2 - axis, (xs.start + xs.stop - 1) / 2 - axis)
            regions.append(BlockedRegion(np.where(labels[ys, xs] == k, rest[:, ys, xs], 0.), centre, M))
    return regions


def raster_region(opaque, d_focal, axis, wavelengths, d_pupil, n_pupil, band_limited=True, square_cells=False, margin=10.,
                  device=None):
    """Blocked region of a raster of opaque cells, clear outside.

    Parameters
    ----------
    opaque : array_like of bool
        True where the mask is opaque, shape (n_y, n_x), rows along y.
    d_focal : float
        Spatial sampling interval of the raster on the sky [rad].
    axis : tuple of float
        Position (y, x) of the optical axis in the raster [samples].
    wavelengths : array_like
        Wavelengths of the light [m], shape (n_wl,).
    d_pupil : float
        Spatial sampling interval of the pupil field [m].
    n_pupil : int
        Number of pupil samples; the second pupil may have up to 2 n_pupil.
    band_limited : bool, optional
        Low-pass the raster to what the pupil grid sees (fast); False keeps every raster
        sample (slow, the direct sums).
    square_cells : bool, optional
        Treat each raster sample as a uniform square cell rather than a point (band-limited
        only).
    margin : float, optional
        Window margin beyond the raster [wavelength / (n_pupil d_pupil)].
    device : torch.device or str, optional
        Device for the one-time filtering.

    Returns
    -------
    BlockedRegion

    Notes
    -----
    The field returned to the second pupil depends on the mask only through spatial
    frequencies below (pupil separation) / wavelength, so the raster is low-passed there
    (raised cosine to twice that) and sampled finely enough for the sums to stay exact;
    only the truncation at the margin remains.
    """
    n_y, n_x = np.shape(opaque)
    centre = ((n_y - 1) / 2 - axis[0], (n_x - 1) / 2 - axis[1])
    if square_cells and not band_limited:
        raise ValueError('square_cells needs band_limited=True')
    if not band_limited:
        n_fft = np.asarray(wavelengths, dtype=np.float64) / (float(d_pupil) * d_focal)
        return BlockedRegion(np.asarray(opaque, dtype=np.float64)[None], centre, n_fft)
    reach = 1.5 * n_pupil * d_pupil
    extent = max(abs(centre[0]) + (n_y - 1) / 2, abs(centre[1]) + (n_x - 1) / 2) * d_focal
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    n_window = 2 * int(np.ceil(extent * 3 * reach / wavelengths.min() + margin * 3 * reach / (n_pupil * d_pupil))) + 1
    raster = torch.as_tensor(np.asarray(opaque, dtype=np.float64), dtype=torch.complex128, device=device)
    blocked = []
    for wavelength in wavelengths:
        d_out = wavelength / (3 * reach)
        period = n_window * d_out
        cutoff = reach / wavelength
        n_freq = 2 * int(np.ceil(2 * cutoff * period)) + 1
        spectrum = bluestein_fft(raster, n_freq, period / d_focal, inverse=True, centre_in=centre) * d_focal**2
        nu = centred_axis(n_freq, 1 / period, device).abs()
        h = torch.where(nu <= cutoff, 1., 0.5 * (1 + torch.cos(torch.pi * torch.clamp(nu / cutoff - 1, 0, 1))))
        if square_cells:
            h = h * torch.sinc(nu * d_focal)
        blocked.append(bluestein_fft(spectrum * h[:, None] * h[None, :], n_window, n_window).real / period**2)
    return BlockedRegion(torch.stack(blocked).cpu().numpy(), (0., 0.), 3 * reach / d_pupil)


def load_raster_region(path, um_per_arcsec, wavelengths, d_pupil, n_pupil, band_limited=True, square_cells=False, device=None):
    """Blocked region from a mask raster file, with the optical axis at the opaque centroid.

    Parameters
    ----------
    path : str
        .npz with `opaque`, cell size `pitch_um`, first-cell corner `x0_um`, `y0_um`
        and `centroid_um` (x, y), in um at the mask.
    um_per_arcsec : float
        Mask scale on the sky [um/arcsec].
    wavelengths : array_like
        Wavelengths of the light [m], shape (n_wl,).
    d_pupil : float
        Spatial sampling interval of the pupil field [m].
    n_pupil : int
        Number of pupil samples.
    band_limited, square_cells : bool, optional
        See raster_region.
    device : torch.device or str, optional
        Device for the one-time filtering.

    Returns
    -------
    BlockedRegion
    """
    raster = np.load(path)
    d_um = float(raster['pitch_um'])
    # Cell i covers [x0 + i d, x0 + (i + 1) d].
    axis = ((raster['centroid_um'][1] - raster['y0_um']) / d_um - 0.5,
            (raster['centroid_um'][0] - raster['x0_um']) / d_um - 0.5)
    return raster_region(raster['opaque'], d_um / um_per_arcsec * np.pi / (180 * 3600), axis, wavelengths, d_pupil,
                         n_pupil, band_limited, square_cells, device=device)


def make_occulter(fpm, wavelengths, d_pupil, n_pupil, mount=True, fpm_raster=None, um_per_arcsec=593.336,
                  band_limited=True, square_cells=False, device=None, **kwargs):
    """BabinetImageOptic from PyFLIP's mask files, with the spot optionally replaced by a mask raster.

    Parameters
    ----------
    fpm : array_like
        Amplitude transmission on PyFLIP's focal grid, shape (n_wl, M, M).
    wavelengths : array_like
        Wavelengths of the light [m], shape (n_wl,).
    d_pupil : float
        Spatial sampling interval of the pupil field [m].
    n_pupil : int
        Number of pupil samples.
    mount : bool, optional
        Keep the mask's mount regions.
    fpm_raster : str, optional
        Mask raster file replacing the spot (see load_raster_region).
    um_per_arcsec : float, optional
        Mask scale on the sky for the raster [um/arcsec].
    band_limited, square_cells : bool, optional
        Band-limit the raster, with square cells (see raster_region).
    device : torch.device or str, optional
        Device for the raster's one-time filtering.
    **kwargs
        Passed to BabinetImageOptic.

    Returns
    -------
    BabinetImageOptic
    """
    regions = flip_mask_regions(fpm, spot=fpm_raster is None, mount=mount)
    if fpm_raster is not None:
        regions.append(load_raster_region(fpm_raster, um_per_arcsec, wavelengths, d_pupil, n_pupil, band_limited, square_cells,
                                          device))
    return BabinetImageOptic(regions, **kwargs)


class DiffDetector(DiffOpticalElement):
    def __init__(self,args,oversample=2, name=None, num_det_px=80, shift_x=-0.12, shift_y=-0.11, device='cuda'):
        super().__init__()
        # Projects a broadbandwavefront phasor to a detetor, using the MFT algorithm
        self.name = name
        self.oversample = oversample
        self.optic_type = 'detector'

        # Unpack for MFT calculations
        (npixels, wavelengths, true_pixel_scale, psf_npix, psf_pixel_scale, focal_length, shift, pixel, inverse) = args

        self.num_det_px = num_det_px#npixels

        xmats, ymats, mults = [], [], []
        for i in range(len(wavelengths)):
            args = (npixels, wavelengths[i].item(), true_pixel_scale, psf_npix*self.oversample, psf_pixel_scale/self.oversample, focal_length, shift, pixel, inverse)
            x_mat, y_mat, mult = partial_MFT(*args)
            xmats.append(x_mat); ymats.append(y_mat); mults.append(torch.tensor(mult, dtype=torch.float64))
        self.x_mat = nn.Parameter(torch.stack(xmats), requires_grad=False).to(device)
        self.y_mat = nn.Parameter(torch.stack(ymats), requires_grad=False).to(device)
        self.mult = nn.Parameter(torch.stack(mults), requires_grad=False).to(device)

        # Extra detector effects
        self.subpixel_shift = DetSubPixShift(shift_x, shift_y)

        # 0th order brighter-fatter effect
        if self.oversample!=1:
            kernel_sigma = 0.28 * self.oversample
            kernel_size = 2 * int(np.ceil(4.0 * kernel_sigma)) + 1   # odd, so the kernel stays centred

            self.charge_diffusion = LearnableGaussianBlur(kernel_size=kernel_size, init_sigma=kernel_sigma)
        else:
            self.charge_diffusion = LearnableGaussianBlur()

        # Flat-fields
        self.detector_flat_field = FlatFieldingModule(self.num_det_px,self.num_det_px, device=device)

    def forward(self,phasors, broadbandwavefront):
        
        # The phasors do the math; broadbandwavefront provides some attributes

        phasor = (self.y_mat.transpose(-2, -1) @ phasors) @ self.x_mat

        phasor *= self.mult.view(1, 1, -1, 1, 1)
        w = (broadbandwavefront.peak_flux) ** 0.5
        out = (torch.abs(phasor) * w) ** 2 

        output = out * broadbandwavefront.wl_weights.view(1, 1, -1, 1, 1)

        output = self.charge_diffusion(output)

        if self.oversample != 1:
            output = torch.sum(torch.reshape(output, (1,len(broadbandwavefront.wl_weights),self.num_det_px,self.oversample,self.num_det_px,self.oversample)), (1,3,5))

        output = broadbandwavefront.flux_correction(output)

        output = self.subpixel_shift(output)

        output = self.detector_flat_field(output)

        return output
    
class DiffOpticalSystem(nn.Module):
    def __init__(self, name=None, optical_element_list = None, oversample = 2):
        super().__init__()
        self.name = name 
        # We want parameters to be in a dictionary that preserves key order.
        # Unnamed elements are keyed by class name and position.
        keys = [(getattr(element, 'name', None) or f'{type(element).__name__}{i}').replace(" ", "")
                for i, element in enumerate(optical_element_list)]
        if sys.version_info >= (3, 7):
            # Can use normal dictionary as it preserves key order
            element_dict = dict(zip(keys, optical_element_list))
        else:
            # have to use an OrderedDict
            from collections import OrderedDict
            element_dict = OrderedDict(zip(keys, optical_element_list))
            print("Older than Python 3.7")
            

        #self.optical_system = nn.ModuleList(optical_element_list) # Assumes first optic is a pupil plane and last is a detector; TODO: make more flexible later
        self.optical_system = nn.ModuleDict(element_dict)
        self.oversample = oversample # Can in principle be different from the detector oversampling

    @property
    def free_parameters(self):
        return dict(self.named_parameters())

    def forward(self, broadbandwavefront):
        phasor = broadbandwavefront.get_phasors()
        normalize=True
        keys = list(self.optical_system.keys())
        for i, key in enumerate(keys):
            element = self.optical_system[key]
            if element.optic_type == 'pupil':
                if isinstance(element, DiffLyotOptic):
                    phasor = element(phasor, broadbandwavefront.wavelengths, normalize=normalize,
                                     d_pupil=broadbandwavefront.pixel_scale)
                else:
                    phasor = element(phasor, broadbandwavefront.wavelengths, normalize=normalize)
                normalize = False # Only normalize at the telescope aperture, first pupil optic
            elif element.optic_type == 'image':
                next_element = self.optical_system[keys[i + 1]] if i + 1 < len(keys) else None
                # A displaced Lyot stop needs the field outside the nominal pupil.
                full_pupil = isinstance(next_element, DiffLyotOptic) and next_element.axial_offset is not None
                phasor = element(phasor,broadbandwavefront.npixels,oversample=self.oversample,
                                 wavelengths=broadbandwavefront.wavelengths,
                                 d_pupil=broadbandwavefront.pixel_scale, crop_output=not full_pupil)
            elif element.optic_type == 'detector':
                detector_image = element(phasor, broadbandwavefront)

        return detector_image
