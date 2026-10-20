"""Centred Fourier sums using Bluestein convolution."""
from functools import lru_cache

import torch
from scipy.fft import next_fast_len


def _chirps(n_in, n_out, n_fft, device, dtype):
    """Calculate the input/output chirps and FFT of the convolution kernel."""
    x = torch.arange(n_in, device=device, dtype=torch.float64) - (n_in - 1) / 2
    u = torch.arange(n_out, device=device, dtype=torch.float64) - (n_out - 1) / 2
    lag = torch.arange(1 - n_in, n_out, device=device, dtype=torch.float64)
    lag = lag + (n_in - n_out) / 2
    chirp_in = torch.exp(-1j * torch.pi * x.square() / n_fft).to(dtype)
    chirp_out = torch.exp(-1j * torch.pi * u.square() / n_fft).to(dtype)
    h = torch.exp(1j * torch.pi * lag.square() / n_fft).to(dtype)
    h_fft = torch.fft.fft(h, n=next_fast_len(n_in + n_out - 1))
    return chirp_in, chirp_out, h_fft


def _axis_factors(n_in, n_out, n_fft, inverse, centre_in, centre_out, device, dtype):
    """Input factor, output factor and kernel FFT for one axis, window centres included."""
    sign = -1 if inverse else 1
    chirp_in, chirp_out, h_fft = _chirps(n_in, n_out, -sign * n_fft, device, dtype)
    x = torch.arange(n_in, device=device, dtype=torch.float64) - (n_in - 1) / 2
    u = torch.arange(n_out, device=device, dtype=torch.float64) - (n_out - 1) / 2
    # A shifted window is a linear phase on the other grid.
    ramp_in = torch.exp(sign * 2j * torch.pi * centre_out * x / n_fft).to(dtype)
    ramp_out = torch.exp(sign * 2j * torch.pi * centre_in * (u + centre_out) / n_fft).to(dtype)
    return chirp_in * ramp_in, chirp_out * ramp_out, h_fft


_fixed_axis_factors = lru_cache(maxsize=64)(_axis_factors)


def bluestein_fft(field, n_out, n_fft, inverse=False, centre_in=(0., 0.), centre_out=(0., 0.)):
    """Compute a zoomed 2D Fourier transform using the Bluestein algorithm.

    Parameters
    ----------
    field : torch.Tensor
        Complex input field, shape (..., n_in_y, n_in_x).
    n_out : int or tuple of int
        Number of output samples, in each dimension or as (n_y, n_x).
    n_fft : float or torch.Tensor
        Equivalent zero-padded length of the input signal. May be noninteger;
        a negative value reverses the transform sign. Must be nonzero.
    inverse : bool, optional
        Use a negative exponent instead of PyFLIP's positive forward exponent.
    centre_in, centre_out : tuple of float, optional
        Centre (y, x) of the input and output windows [samples]; (0, 0) is the
        optical axis.

    Returns
    -------
    torch.Tensor
        Unnormalized Fourier transform, shape (..., n_y, n_x).

    Notes
    -----
    Coordinates are centred on (n - 1)/2 plus the window centre. The transform runs
    along x and then y, each axis cropped to its output before the next. The
    equivalent zero-padded array is not allocated. Chirps are cached for fixed
    sampling and recalculated when gradients through the sampling are required.
    """
    n_out = (n_out, n_out) if isinstance(n_out, int) else tuple(n_out)
    trainable = torch.is_tensor(n_fft) and n_fft.requires_grad
    factors = _axis_factors if trainable else _fixed_axis_factors
    n_fft = n_fft if trainable else float(n_fft)
    for dim, axis in ((-1, 1), (-2, 0)):
        field = field.movedim(dim, -1)
        n_in = field.shape[-1]
        factor_in, factor_out, h_fft = factors(n_in, n_out[axis], n_fft, inverse, float(centre_in[axis]),
                                               float(centre_out[axis]), field.device, field.dtype)
        convolution = torch.fft.ifft(torch.fft.fft(field * factor_in, n=h_fft.numel()) * h_fft)
        field = (convolution[..., n_in - 1:n_in - 1 + n_out[axis]] * factor_out).movedim(-1, dim)
    return field


def centered_fft(field, inverse=False):
    """Compute a 2D FFT with PyFLIP's coordinate and sign conventions.

    Parameters
    ----------
    field : torch.Tensor
        Complex input field, shape (..., n, n), centered on (n - 1)/2.
    inverse : bool, optional
        Use a negative exponent instead of the positive forward exponent.

    Returns
    -------
    torch.Tensor
        Unnormalized Fourier transform on the same size grid.
    """
    n = field.shape[-1]
    axis = torch.arange(n, device=field.device, dtype=torch.float64)
    centre = (n - 1) / 2
    sign = -1 if inverse else 1
    phase_factor = torch.exp(-sign * 2j * torch.pi * centre * (axis - centre / 2) / n).to(field.dtype)
    input_field = field * phase_factor[:, None] * phase_factor[None, :]
    result = torch.fft.fft2(input_field) if inverse else torch.fft.ifft2(input_field) * n**2
    return result * phase_factor[:, None] * phase_factor[None, :]
