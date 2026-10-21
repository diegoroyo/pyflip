"""Null-space walk of fitted PyFLIP OPD maps.

A fitted pupil (JWSTPrimary) or Lyot-plane (NIRCamLyotstop) OPD map is not unique: many maps
give the same detector image, and an optimiser leaves an arbitrary part of the map in
directions the data cannot see. The walk removes that part step by step while the model's
image stays unchanged (star position and flux are free) and the detector window keeps its
share of the light after the Lyot stop, so starlight cannot be hidden outside the window.

USAGE

    python null_space_walk.py --state fitted_reference.npz --steps 30

        a state saved by flip_stages_fixed_geometry.py (maps, flat field, star, flux, geometry
        and the options that rebuild its model);

    python null_space_walk.py --pupil_map pupil.npy --lyot_map lyot.npy --model_dir <dir> \\
        --data <calints.fits> --px_mask pixel_masks.npy [--star_mas x,y] --steps 30

        maps fitted with PyFLIP's default model of a model directory (.npy, metres); without
        --star_mas the star position is fitted to the data first.

OUTPUT

<out>.npz with the walked maps (same fields as --state; star and flux refitted to the start
image) and <out>.log, one line per trial step: map RMS over the pupil and stop openings, RMS
of the part removed a fraction at a time (invisible), image change sum(((image - start) /
noise)^2) against the budget (default sqrt(2 N) for N pixels), window share relative to the
start, and chi^2/N against the data. If the conjugate-gradient residual grows well above the
tolerance the invisible part is inaccurate and the maps stop shrinking: raise --rank or
--n_iter.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator, cg
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'refactoring'))

MAS = np.radians(1 / 3.6e6)                                   # [rad]


#### WALK
def jacobian_products(residual, theta):
    """Residual and Jacobian products of a residual function, from reverse mode only.

    Parameters
    ----------
    residual : callable
        Parameters, shape (n,) -> residual, shape (m,).
    theta : torch.Tensor
        Parameters, shape (n,).

    Returns
    -------
    r : torch.Tensor
        Residual at theta.
    jvp, vjp : callable
        v -> J v and u -> J^T u at theta.
    """
    r, vjp_fn = torch.func.vjp(residual, theta)
    vjp = lambda u: vjp_fn(u)[0]
    jvp = lambda v: torch.func.vjp(vjp, torch.zeros_like(r))[1](v)[0].detach()
    return r.detach(), jvp, lambda u: vjp(u).detach()


def nuisance_projector(columns):
    """u -> u - Q Q^T u, with Q an orthonormal basis of the residual derivatives along the nuisances, shape (m, k)."""
    q = torch.linalg.qr(columns)[0]
    return lambda u: u - q @ (q.T @ u)


def nystrom_preconditioner(apply_K, m, rank, delta, like):
    """Randomized Nystrom preconditioner of K + delta I, K symmetric positive semi-definite.

    Parameters
    ----------
    apply_K : callable
        x -> K x, shape (m,).
    m, rank : int
        Size of K and rank of the approximation.
    delta : float
        Shift.
    like : torch.Tensor
        Gives the dtype and device.

    Returns
    -------
    callable
        x -> P^-1 x = (lam_k + delta) U (Lam + delta I)^-1 U^T x + (I - U U^T) x.

    Notes
    -----
    Frangella, Tropp and Udell, SIAM J. Matrix Anal. Appl. 44, 718 (2023).
    """
    omega = torch.linalg.qr(torch.randn(m, rank, dtype=like.dtype, device=like.device))[0]
    Y = torch.stack([apply_K(omega[:, j]) for j in range(rank)], 1)
    nu = torch.finfo(like.dtype).eps * m**0.5 * float(Y.norm())
    Y = Y + nu * omega
    B = torch.linalg.solve_triangular(torch.linalg.cholesky(omega.T @ Y), Y.T, upper=False).T
    U, sv, _ = torch.linalg.svd(B, full_matrices=False)
    lam = torch.clamp(sv**2 - nu, min=0.)
    scale = (lam[-1] + delta) / (lam + delta)
    return lambda x: U @ (scale * (U.T @ x)) + x - U @ (U.T @ x)


def minimum_norm_change(jvp, vjp, project, support, b, delta, n_iter, tol, apply_Minv=None):
    """Smallest supported parameter change whose first-order projected residual change is b.

    dtheta = S J^T W y with (W J S J^T W + delta I) y = W b, for the support S and the nuisance
    projector W, solved by scipy's conjugate gradients. Directions with squared singular value
    below delta count as invisible. Returns dtheta, the iterations and the final relative residual.
    """
    to_torch = lambda x: torch.as_tensor(x, dtype=b.dtype, device=b.device)
    numpy_op = lambda f: (lambda x: f(to_torch(x)).cpu().numpy())
    A = LinearOperator((len(b), len(b)), matvec=numpy_op(lambda y: project(jvp(support * vjp(project(y)))) + delta * y))
    M = None if apply_Minv is None else LinearOperator((len(b), len(b)), matvec=numpy_op(apply_Minv))
    rhs = project(b).cpu().numpy()
    iterations = [0]
    y, _ = cg(A, rhs, rtol=tol, maxiter=n_iter, M=M, callback=lambda _: iterations.__setitem__(0, iterations[0] + 1))
    rel = np.linalg.norm(rhs - A @ y) / max(np.linalg.norm(rhs), 1e-300)
    return support * vjp(project(to_torch(y))), iterations[0], rel


def null_space_walk(residual, nuisance_columns, theta, support, steps, delta, budget, step_fraction=0.1,
                    retractions=4, n_iter=300, tol=1e-3, rank=0, report=None):
    """Remove the invisible part of the supported parameters while the projected residual stays within budget.

    Parameters
    ----------
    residual : callable
        Parameters, shape (n,) -> residual from the start state, shape (m,); zero at the start.
    nuisance_columns : callable
        Parameters -> residual derivatives along the nuisance parameters, shape (m, k).
    theta, support : torch.Tensor
        Start parameters, and 1 on those walked, 0 elsewhere, shape (n,).
    steps : int
        Accepted steps.
    delta : float
        Tikhonov parameter separating the invisible part.
    budget : float
        Largest squared norm of the projected residual of an accepted state.
    step_fraction : float, optional
        Largest fraction of the invisible part removed in one step.
    retractions : int, optional
        Gauss-Newton steps pulling each trial back towards the start residual.
    n_iter, tol : optional
        Conjugate-gradient iteration limit and relative tolerance.
    rank : int, optional
        Rank of the Nystrom preconditioner, rebuilt when a solve misses tol; 0: none.
    report : callable, optional
        Called after each trial with (accepted, trial parameters, info dict).

    Returns
    -------
    theta : torch.Tensor
        Walked parameters.
    history : list of dict
        Info of the accepted steps.
    """
    def linearise(t):
        r, jvp, vjp = jacobian_products(residual, t)
        project = nuisance_projector(nuisance_columns(t))
        return project(r), jvp, vjp, project

    def cost(t, project):
        with torch.no_grad():
            r = project(residual(t))
        return float(r @ r)

    def preconditioner(jvp, vjp, project, r):
        if not rank:
            return None
        return nystrom_preconditioner(lambda y: project(jvp(support * vjp(project(y)))), len(r), rank, delta, r)

    history, a = [], step_fraction
    r, jvp, vjp, project = linearise(theta)
    Minv = preconditioner(jvp, vjp, project, r)
    while len(history) < steps and a > step_fraction / 64:
        t0 = time.time()
        s = support * theta
        visible, n, rel = minimum_norm_change(jvp, vjp, project, support, jvp(s), delta, n_iter, tol, Minv)
        if rank and rel > tol:
            Minv = preconditioner(jvp, vjp, project, r)
            visible, n, rel = minimum_norm_change(jvp, vjp, project, support, jvp(s), delta, n_iter, tol, Minv)
        invisible = s - visible
        trial = theta - a * invisible
        r_t, jvp_t, vjp_t, project_t = linearise(trial)
        for _ in range(retractions):
            dtheta, _, _ = minimum_norm_change(jvp_t, vjp_t, project_t, support, r_t, delta, n_iter, tol, Minv)
            current = float(r_t @ r_t)
            damping = next((c for c in (1., 0.5, 0.25, 0.1) if cost(trial - c * dtheta, project_t) < current), None)
            if damping is None:
                break
            trial = trial - damping * dtheta
            r_t, jvp_t, vjp_t, project_t = linearise(trial)
        info = dict(step_fraction=a, invisible=float(invisible.norm()), cost=float(r_t @ r_t), cg=n, cg_residual=rel,
                    time=time.time() - t0)
        accepted = info['cost'] <= budget
        if report is not None:
            report(accepted, trial, info)
        if accepted:
            theta, r, jvp, vjp, project = trial, r_t, jvp_t, vjp_t, project_t
            history.append(info)
            a = min(step_fraction, 1.5 * a)
        else:
            a = a / 2
    return theta, history


#### PYFLIP MODEL
PUPIL = 'instrument.optical_system.JWSTPrimary.wfe_offsets.grid'
LYOT = 'instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid'
STAR = 'wavefront.angles_offset'


def state_from_args(args):
    """Fitted state from --state, or from PyFLIP map files with the model directory's default model."""
    if args.state:
        z = np.load(args.state)
        state = {k: z[k] for k in z.files if k != 'options'}
        state['options'] = json.loads(str(z['options']))
        return state
    read_map = lambda path: np.load(path).astype(np.float64).reshape(1024, 1024)
    star = None if args.star_mas is None else MAS * np.array([float(v) for v in args.star_mas.split(',')])
    options = dict(model_dir=args.model_dir, data_file=args.data, px_mask=args.px_mask, num_det_px=args.num_det_px)
    return dict(pupil=read_map(args.pupil_map), lyot=read_map(args.lyot_map), star=star, options=options)


def build_model(state, device):
    """PyFLIP instrument and wavefront of a fitted state; the maps and the star are set at each call."""
    from wavefronts import BroadbandWavefront
    from model_building import assemble_JWST_NIRCam_coron
    from dl_utils import load_model_data
    o = state['options']
    _, _, _, _, wlen_weights, _ = load_model_data(o['model_dir'], substrate=o.get('substrate', True))
    instrument = assemble_JWST_NIRCam_coron(o['model_dir'], o['num_det_px'], DEVICE=device,
                                            babinet=o.get('fpm_raster') is not None or not o.get('mount', True),
                                            mount=o.get('mount', True), fpm_raster=o.get('fpm_raster'),
                                            um_per_arcsec=o.get('um_per_arcsec') or 593.336, band_limited=o.get('band_limited', True),
                                            square_cells=o.get('square_cells', False),
                                            fpm_axial_offset=float(state.get('fpm_dz', 0.)) or None, fpm_focal_length=131.4)
    lyot, detector = instrument.optical_system.NIRCamLyotstop, instrument.optical_system.NIRCamdetector
    if o.get('lyot_file'):
        lyot.amplitude = torch.FloatTensor(np.load(o['lyot_file'])).to(device)[None]
    if state.get('registration') is not None:
        detector.subpixel_shift.shift_x, detector.subpixel_shift.shift_y = (float(v) for v in state['registration'])
    with torch.no_grad():
        if state.get('lyot_shift') is not None:
            lyot.pupil_shifts.x_shift.fill_(float(state['lyot_shift'][0]))
            lyot.pupil_shifts.y_shift.fill_(float(state['lyot_shift'][1]))
            lyot.pupil_shifts.rotation.fill_(float(state['clocking']))
        if state.get('blur') is not None:
            detector.charge_diffusion.log_sigma.fill_(np.log(2 * float(state['blur'])))
        if state.get('flat') is not None:
            detector.detector_flat_field.grid.copy_(torch.as_tensor(state['flat']).reshape(detector.detector_flat_field.grid.shape))
    wavefront = BroadbandWavefront(1024, 6.603464, wlen_weights, torch.ones(1, dtype=torch.float64),
                                   angles_offset=torch.zeros(2, dtype=torch.float64), DEVICE=device).to(device)
    module = nn.ModuleDict({'instrument': instrument, 'wavefront': wavefront}).requires_grad_(False)
    module.forward = lambda: instrument(wavefront)
    return module


def load_frame(options, device):
    """Data and noise on the used pixels, and those pixels' flat indices."""
    from dl_utils import load_and_crop_data
    data, noise = (np.squeeze(a).astype(np.float64) for a in load_and_crop_data(options['data_file'],
                                                                               num_det_px=options['num_det_px']))
    good = np.isfinite(data) & np.isfinite(noise) & (noise > 0)
    if options.get('px_mask'):
        good &= np.squeeze(np.load(options['px_mask'])) > 0
    pixels = torch.as_tensor(np.flatnonzero(good), device=device)
    return (torch.as_tensor(data, device=device).reshape(-1)[pixels], torch.as_tensor(noise, device=device).reshape(-1)[pixels],
            pixels)


def fit_star(image_of, star, target, noise):
    """Star offset [rad] fitted by least squares to a target image, the flux solved at each offset; returns star, flux."""
    def flux_and_residual(star_mas):
        image = image_of(torch.as_tensor(star_mas * MAS, dtype=target.dtype, device=target.device))
        flux = (image * target / noise**2).sum() / (image**2 / noise**2).sum()
        return float(flux), ((flux * image - target) / noise).cpu().numpy()
    star_mas = least_squares(lambda s: flux_and_residual(s)[1], np.asarray(star.cpu()) / MAS, diff_step=1e-3).x
    return torch.as_tensor(star_mas * MAS, dtype=target.dtype, device=target.device), flux_and_residual(star_mas)[0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--state', help='state saved by flip_stages_fixed_geometry.py (fitted_*.npz)')
    parser.add_argument('--pupil_map', help='fitted JWSTPrimary OPD offsets (.npy, m)')
    parser.add_argument('--lyot_map', help='fitted NIRCamLyotstop OPD offsets (.npy, m)')
    parser.add_argument('--model_dir', help='PyFLIP model directory the maps were fitted with')
    parser.add_argument('--data', help='data frame the maps were fitted to (calints FITS or coadded .npy)')
    parser.add_argument('--px_mask', help='pixel mask (.npy, 1 = used)')
    parser.add_argument('--num_det_px', type=int, default=80, help='detector window [pixels]')
    parser.add_argument('--star_mas', help='fitted star offset x,y [mas]; default: fitted to the data')
    parser.add_argument('--steps', type=int, default=30, help='accepted steps')
    parser.add_argument('--step_fraction', type=float, default=0.1, help='largest fraction of the invisible part removed per step')
    parser.add_argument('--budget', type=float, help='largest image change, sum of ((image - start) / noise)^2; default sqrt(2 N)')
    parser.add_argument('--delta', type=float, help='Tikhonov parameter; default budget / (4 x squared start map norm)')
    parser.add_argument('--share_weight', type=float, default=300., help='weight of the window-share row: the relative share '
                        'may change by up to sqrt(budget) / share_weight')
    parser.add_argument('--rank', type=int, default=800, help='rank of the Nystrom preconditioner; 0: none')
    parser.add_argument('--n_iter', type=int, default=300, help='conjugate-gradient iterations per solve')
    parser.add_argument('--out', help='output name without extension; default: <input>_walked')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if not args.state and not (args.pupil_map and args.lyot_map and args.model_dir and args.data):
        parser.error('give --state, or --pupil_map, --lyot_map, --model_dir and --data')
    return args


def main():
    args = parse_args()
    out = args.out or os.path.splitext(args.state or args.pupil_map)[0] + '_walked'
    log = open(out + '.log', 'w')

    def say(line):
        print(line, flush=True)
        log.write(line + '\n')

    state = state_from_args(args)
    module = build_model(state, args.device)
    system = module['instrument'].optical_system
    post_stop = {}
    system.NIRCamLyotstop.register_forward_hook(lambda m, i, output: post_stop.update(field=output))
    data, noise, pixels = load_frame(state['options'], args.device)

    shape = system.JWSTPrimary.wfe_offsets.grid.shape
    n_map = int(np.prod(shape))
    support = torch.cat([system.JWSTPrimary.amplitude.reshape(-1) > 0,
                         system.NIRCamLyotstop.transmission().detach().reshape(-1) > 1e-3]).double()
    unit = 1e-9 * float(support.sum())**0.5                    # a parameter vector of norm 1: maps of 1 nm RMS [m]
    theta0 = torch.as_tensor(np.concatenate([state['pupil'].reshape(-1), state['lyot'].reshape(-1)]), device=args.device) / unit

    def render(theta, star):
        """Model image on the used pixels, and the window share of the post-stop light."""
        params = {PUPIL: (unit * theta[:n_map]).reshape(shape), LYOT: (unit * theta[n_map:]).reshape(shape), STAR: star}
        image = torch.func.functional_call(module, params, ()).reshape(-1)
        energy = (post_stop['field'].abs()**2 * module['wavefront'].wl_weights.view(-1, 1, 1)).sum()
        return image[pixels], image.sum() / energy

    image_of = lambda theta: (lambda star: render(theta, star)[0].detach())
    zeros = torch.zeros(2, dtype=torch.float64, device=args.device)
    star0 = (torch.as_tensor(state['star'], device=args.device) if state.get('star') is not None
             else fit_star(image_of(theta0), zeros, data, noise)[0])
    with torch.no_grad():
        image0, share0 = render(theta0, star0)
    flux0 = float((image0 * data / noise**2).sum() / (image0**2 / noise**2).sum())
    budget = args.budget if args.budget is not None else (2 * len(pixels))**0.5
    delta = args.delta if args.delta is not None else budget / (4 * float((support * theta0).norm())**2)

    def residual_with(theta, nuisance):
        """Whitened image change and weighted window-share change; nuisance = (star x, star y, flux scale)."""
        image, share = render(theta, nuisance[:2])
        return torch.cat([flux0 * (nuisance[2] * image - image0) / noise, args.share_weight * (share / share0 - 1)[None]])

    nuisance0 = torch.cat([star0, torch.ones(1, dtype=torch.float64, device=args.device)])
    residual = lambda theta: residual_with(theta, nuisance0)

    def nuisance_columns(theta):
        _, jvp, _ = jacobian_products(lambda q: residual_with(theta, q), nuisance0)
        return torch.stack([jvp(e) for e in torch.eye(3, dtype=torch.float64, device=args.device)], 1)

    def chi2(theta):
        image = image_of(theta)(star0)
        flux = (image * data / noise**2).sum() / (image**2 / noise**2).sum()
        return float((((flux * image - data) / noise)**2).mean())

    rms = lambda theta: float((support * theta).norm())
    share_of = lambda theta: float(render(theta, star0)[1].detach() / share0)
    chi2_0 = chi2(theta0)
    say('%s | %d pixels | budget %.1f | delta %.3g | share weight %g' % (os.path.basename(state['options']['data_file']), len(pixels),
                                                                          budget, delta, args.share_weight))
    say('start: maps %.1f nm RMS | chi2/N %.4f' % (rms(theta0), chi2_0))
    n_accepted = [0]

    def report(accepted, theta, info):
        n_accepted[0] += accepted
        say('%s a %.3f: maps %.1f nm RMS, invisible %.1f nm, image change %.1f (budget %.0f), window share %.4f, chi2/N %.4f, '
            'CG %d it (residual %.1e), %.1f s' % ('step %d' % n_accepted[0] if accepted else 'rejected', info['step_fraction'],
                                                  rms(theta), info['invisible'], info['cost'], budget, share_of(theta), chi2(theta),
                                                  info['cg'], info['cg_residual'], info['time']))

    t0 = time.time()
    theta, history = null_space_walk(residual, nuisance_columns, theta0, support, args.steps, delta, budget, args.step_fraction,
                                     n_iter=args.n_iter, rank=args.rank, report=report)
    star, flux = fit_star(image_of(theta), star0, flux0 * image0, noise)
    say('walked: %d steps in %.0f s | maps %.1f -> %.1f nm RMS | chi2/N %.4f -> %.4f | window share %.4f' % (
        len(history), time.time() - t0, rms(theta0), rms(theta), chi2_0, chi2(theta), share_of(theta)))

    saved = {k: v for k, v in state.items() if k != 'options' and v is not None}
    saved.update(pupil=(unit * theta[:n_map]).reshape(1024, 1024).cpu().numpy(), lyot=(unit * theta[n_map:]).reshape(1024, 1024).cpu().numpy(),
                 star=star.cpu().numpy(), flux=flux)
    np.savez(out + '.npz', options=json.dumps(state['options']), **saved)
    say('saved %s.npz and %s.log' % (out, out))


if __name__ == '__main__':
    main()
