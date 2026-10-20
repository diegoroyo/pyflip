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


@lru_cache(maxsize=32)
def _fixed_chirps(n_in, n_out, n_fft, device, dtype):
    """Retrieve cached chirps for fixed input and output sampling."""
    return _chirps(n_in, n_out, n_fft, device, dtype)


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
    Coordinates are centred on (n - 1)/2 plus the window centre. The equivalent
    zero-padded array is not allocated. Chirps are cached for fixed sampling
    and recalculated when gradients through the sampling are required.
    """
    n_in = tuple(field.shape[-2:])
    n_out = (n_out, n_out) if isinstance(n_out, int) else tuple(n_out)
    scale = (-1 if inverse else 1) / n_fft
    # A shifted window is a linear phase on the other grid.
    if any(centre_out):
        field = field * _linear_phase(n_in, centre_out, scale, field)
    n_fft = n_fft if inverse else -n_fft
    if torch.is_tensor(n_fft) and n_fft.requires_grad:
        (in_y, out_y, h_y), (in_x, out_x, h_x) = [_chirps(i, o, n_fft, field.device, field.dtype) for i, o in zip(n_in, n_out)]
    else:
        (in_y, out_y, h_y), (in_x, out_x, h_x) = [_fixed_chirps(i, o, float(n_fft), field.device, field.dtype)
                                                  for i, o in zip(n_in, n_out)]
    field_fft = torch.fft.fft2(field * in_y[:, None] * in_x[None, :], s=(h_y.numel(), h_x.numel()))
    convolution = torch.fft.ifft2(field_fft * h_y[:, None] * h_x[None, :])
    result = convolution[..., n_in[0] - 1:n_in[0] - 1 + n_out[0], n_in[1] - 1:n_in[1] - 1 + n_out[1]]
    result = result * out_y[:, None] * out_x[None, :]
    if any(centre_in):
        result = result * _linear_phase(n_out, centre_in, scale, field, centre_out)
    return result


def _linear_phase(shape, slope, scale, like, origin=(0., 0.)):
    """exp(2 pi i scale (slope_y (y + origin_y) + slope_x (x + origin_x))) on a centred grid of the given shape."""
    y, x = [torch.arange(n, device=like.device, dtype=torch.float64) - (n - 1) / 2 for n in shape]
    phase = slope[0] * (y[:, None] + origin[0]) + slope[1] * (x[None, :] + origin[1])
    return torch.exp(2j * torch.pi * scale * phase).to(like.dtype)


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
