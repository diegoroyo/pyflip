import torch
import torch.nn as nn
import torch.nn.functional as F

# Band-limited free OPD grid: a drop-in for model_classes.OPDOffsetModule (same 'grid' parameter and forward(opd)) whose
# parameters are the nodes of a coarse n_nodes x n_nodes grid upsampled to the optic's npixels x npixels. upsample =
# 'fourier': zero-padding of the nodes' spectrum, cut to a disk, so the layer contains exactly nothing above n_nodes / 2
# cycles per array in any direction (periodic over the array; the optic only uses its centre); 'bicubic': torch's bicubic interpolation, which lets some of
# the node pattern's power through above that (a fitted 24-node pattern leaked 15% of its rms above 18 cycles per pupil).

class CoarseOPDGrid(nn.Module):
    def __init__(self, n_nodes, npixels, upsample='fourier', DEVICE='cuda'):
        super().__init__()
        self.npixels = npixels
        self.upsample = upsample
        self.grid = nn.Parameter(torch.zeros((1, n_nodes, n_nodes), device=DEVICE))

    def get_res(self):
        if self.upsample == 'bicubic':
            return F.interpolate(self.grid[None], size=(self.npixels, self.npixels), mode='bicubic', align_corners=False)[0]
        m, n = self.grid.shape[-1], self.npixels
        spectrum = torch.fft.fftshift(torch.fft.fft2(self.grid), dim=(-2, -1))
        padded = torch.zeros((1, n, n), dtype=spectrum.dtype, device=spectrum.device)
        c, h = n // 2, m // 2
        k = torch.arange(m, device=spectrum.device) - h
        padded[:, c - h:c - h + m, c - h:c - h + m] = spectrum * ((k[:, None] ** 2 + k[None, :] ** 2) <= h ** 2)
        return torch.fft.ifft2(torch.fft.ifftshift(padded, dim=(-2, -1))).real * (n / m) ** 2

    def forward(self, x):
        return self.get_res() + x


# Star flux with a learnable relative scale: a drop-in for model_classes.FluxOffsetModule (wavefronts.BroadbandWavefront.
# flux_correction). 'data' is the factor that dl_utils.hardcoded_match_flux rescales to the data in stages 1.1 and 2.1
# (about 1e7 here, so a gradient step on it would be negligible); 'log_scale' is fitted by gradient on top of it, an Adam
# step of lr being a relative change of about lr. factor() is exp(log_scale), 1 = the matched flux.

class FluxScale(nn.Module):
    def __init__(self, DEVICE='cuda'):
        super().__init__()
        self.data = nn.Parameter(torch.ones(1, device=DEVICE), requires_grad=False)
        self.log_scale = nn.Parameter(torch.zeros(1, device=DEVICE))

    def factor(self):
        return torch.exp(self.log_scale)

    def forward(self, x):
        return self.data * self.factor() * x
