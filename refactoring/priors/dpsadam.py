import torch
import torch.nn as nn
import numpy as np
from priors.schedule import EDMSchedule

# DPS-Adam (InverseBench algo/dps_adam.py: diffusion posterior sampling where the likelihood step is an Adam step)
# driven by pyFLIP's own optimization loop. DPSAdamOPD replaces model_classes.OPDOffsetModule as the 'wfe_offsets' of
# a DiffPupilOptic: like it, it has a free 'grid' parameter and forward(opd) returns the OPD layer, so the optimizer
# groups and the stage handlers of testing.py work unchanged.
#
# The OTE OPD layer is a sum of pieces, by pyFLIP iteration i:
#   - the optic's own OPD (pyFLIP's WFS map) in mode 'offset', or in mode 'replace' only before the first phase starts
#     (init='wfs'; init='mean' gives the first prior's mean map instead). In mode 'replace' the field term (stpsf's
#     field-dependent OTE WFE, which the WFS map contains and the prior's training maps do not) is added from then on.
#   - one DPSPhase per DPS-Adam sampling window [start, end): during the window, one diffusion step per iteration (the
#     OPD is to_opd(D(latent, sigma)) with the graph attached, pyFLIP's loss.backward() puts the likelihood gradient in
#     latent.grad, pyFLIP's optimizer applies it (the latents are one of its parameter groups) and the prior move of
#     the latent (reverse diffusion step, score from the same denoised map) is applied at the top of the next
#     iteration; the two moves are additive, so this is the same as applying the prior move first, as dps_adam.py).
#     After the window the phase's output is frozen (the latent, or D(latent, sigma_final) if final_denoise).
#     Typical use: one phase on the reference frame with a full-map prior (mode 'replace') or a drift prior
#     (mode 'offset'), optionally a second phase on the science frame with a drift prior (a drift on top of what
#     the reference stage gave). Phases must not overlap.
#   - 'grid', pyFLIP's free OPD, optimized by pyFLIP's own code like OPDOffsetModule.grid; its gradient is zeroed
#     (hook) before grid_frozen_until (default: the end of the first phase) and inside every phase window, so it only
#     moves in between (reference stage) and after the last phase (science stage, see priors/stages.py). Note that
#     default_stages zeroes no gradient at an iteration equal to a stage cutoff, so a grid whose Adam state is still
#     empty would take a large step there; keep it frozen past such iterations.
#
# Use in the loop (see testing_priors.py): ote_prior.step(i) right after optimizer.zero_grad(), nothing else.

class DPSPhase:
    """
    One DPS-Adam sampling window: its prior, iterations [start, end) (end - start diffusion steps), latent and state.
    """
    def __init__(self, prior, start, end, sigma_max=10., sigma_min=0.002, sde=True, final_denoise=False, seed=None, condition=None, DEVICE='cuda'):
        self.prior = prior
        self.start, self.end = start, end
        self.sde = sde # reverse SDE (noise added at each prior move) or probability-flow ODE
        self.final_denoise = final_denoise
        self.condition = condition
        self.schedule = EDMSchedule(end - start, sigma_max=sigma_max, sigma_min=sigma_min)

        if seed is not None:
            torch.manual_seed(seed)
        self.latent = nn.Parameter(torch.randn(prior.latent_shape(), device=DEVICE) * sigma_max) # network units, float32
        self.current_opd = None # to_opd(D(latent, sigma)) of this iteration, with its graph
        self.frozen_opd = None # output after the window
        self.pending = None # (k, latent, denoised) of the last diffusion step, for the prior move at the next step()

    def active(self, i):
        return self.start <= i < self.end

    def step(self, i):
        if self.pending is not None:
            # Prior move of the previous diffusion step (the latent already got its likelihood step from the optimizer)
            k, x_k, denoised = self.pending
            self.pending = None
            sigma, factor = self.schedule.sigma_steps[k], self.schedule.factor_steps[k]
            score = (denoised - x_k) / sigma ** 2
            with torch.no_grad():
                move = factor * score
                if self.sde:
                    move = move + np.sqrt(factor) * torch.randn_like(x_k)
                self.latent.add_(move)

        if self.active(i):
            k = i - self.start
            denoised = self.prior.denoise(self.latent, self.schedule.sigma_steps[k], self.condition)
            self.current_opd = self.prior.to_opd(denoised)
            self.pending = (k, self.latent.detach().clone(), denoised.detach())
        elif i >= self.end and self.frozen_opd is None:
            with torch.no_grad():
                x = self.latent.detach()
                if self.final_denoise:
                    x = self.prior.denoise(x, self.schedule.sigma_steps[-1], self.condition)
                self.frozen_opd = self.prior.to_opd(x)

    def opd(self, i):
        # This phase's piece of the OPD layer at iteration i: None before its window, with the graph during, frozen after
        if i < self.start:
            return None
        return self.current_opd if self.active(i) else self.frozen_opd


class DPSAdamOPD(nn.Module):
    def __init__(self, phases, mode='replace', init='wfs', field_term=None, grid_frozen_until=None, npixels=1024, DEVICE='cuda'):
        super().__init__()
        self.phases = list(phases) # DSPPhase objects, in order of their windows
        self.latents = nn.ParameterList([p.latent for p in self.phases]) # registered so the module owns them
        self.mode = mode # 'replace' (first prior models the full OPD, WFS map ignored) or 'offset' (priors model drifts on the WFS map)
        for n, p in enumerate(self.phases):
            if n > 0:
                assert p.start >= self.phases[n - 1].end, 'DPS phases must not overlap'
            if p.prior.kind == 'map':
                assert n == 0 and mode == 'replace', "a 'map' prior can only be the first phase, in mode 'replace'"
            else:
                assert n > 0 or mode == 'offset', "a 'drift' prior as first phase needs mode 'offset'"
        self.grid_frozen_until = self.phases[0].end if grid_frozen_until is None else grid_frozen_until
        self.grid = nn.Parameter(torch.zeros((1, npixels, npixels), device=DEVICE)) # pyFLIP's free OPD
        self.grid.register_hook(self._freeze_grid)

        # Fixed maps, (1, npixels, npixels) metres, pyFLIP orientation
        self.field_term = torch.zeros((1, npixels, npixels), dtype=torch.float64, device=DEVICE)
        if field_term is not None:
            self.field_term = self.field_term + field_term.to(DEVICE)
        self.init = init # 'wfs': the optic's own OPD before the first phase; 'mean': the first prior's mean map (+ field term)
        self.fixed_opd = self.phases[0].prior.mean_opd() if init == 'mean' else None
        self.iteration = -1

    def _freeze_grid(self, grad):
        frozen = self.iteration < self.grid_frozen_until or any(p.active(self.iteration) for p in self.phases)
        return grad * 0. if frozen else grad

    def param_group(self, lr):
        # Parameter group for pyFLIP's optimizer: the DPS-Adam guidance step on the latents, lr = guidance scale
        return {'params': list(self.latents), 'lr': lr}

    def step(self, i):
        # Call once per pyFLIP iteration, before the forward model is evaluated
        self.iteration = i
        for p in self.phases:
            p.step(i)

    def get_res(self):
        # Prior part of the OPD layer (without the optic's own OPD and the grid); None before the first phase with init='wfs'
        pieces = [p.opd(self.iteration) for p in self.phases]
        pieces = [o for o in pieces if o is not None]
        if len(pieces) == 0:
            return None if self.fixed_opd is None else self.fixed_opd + self.field_term
        return sum(pieces) + self.field_term

    def forward(self, x):
        res = self.get_res()
        if res is None:
            return x + self.grid # before the first phase, init='wfs': as OPDOffsetModule with a zero grid
        if self.mode == 'offset':
            res = res + x
        return res + self.grid
