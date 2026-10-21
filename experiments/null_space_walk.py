"""Null-space walk of fitted PyFLIP OPD maps.

A fitted pupil (JWSTPrimary) or Lyot-plane (NIRCamLyotstop) OPD map is not unique: many maps
give the same detector images, and an optimiser leaves an arbitrary part of the map in
directions the data cannot see. The walk removes that part step by step while the model's
images of the given frames stay unchanged (star position and flux are free nuisances) and
each frame keeps its share of the light after the Lyot stop inside the detector window, so
the walk cannot hide starlight outside the window. What is left is the part of the maps the
images need.

USAGE

1. From a state saved by flip_stages_fixed_geometry.py (fitted_reference.npz or
   fitted_final.npz: maps, flat field, star, flux, geometry and the options that rebuild the
   model):

       python null_space_walk.py --state fitted_reference.npz --steps 30

2. From PyFLIP's own fitted maps (.npy, metres, 1024 x 1024; any leading axes of length 1
   are dropped), with the model directory and the data frame they were fitted to:

       python null_space_walk.py --pupil_map pupil_opd.npy --lyot_map lyot_opd.npy \\
           --model_dir HIP65426/model_setup_newpath --data ref_calints.fits \\
           --px_mask pixel_masks.npy --star_mas 12.3,-4.1 --steps 30

   Without --star_mas the star position is fitted to the data first (Gauss-Newton); the
   flux is always solved by least squares. The model is PyFLIP's assemble_JWST_NIRCam_coron:
   its mask files and Lyot stop as stored, unless --fpm_raster / --no_mount / --lyot_file
   and the geometry options (--registration, --blur, --lyot_shift, --clocking, --fpm_dz)
   say otherwise. --flat gives a fitted detector flat field.

   --frames adds further data frames (e.g. other dithers) whose images are kept as well;
   their star positions are fitted first.

Run from refactoring/ or anywhere (the modules are found relative to this file); a GPU is
used when there is one (--device). Data frames are PyFLIP's calints FITS files or its
coadded .npy frames (see dl_utils.load_and_crop_data).

OUTPUT

<out>.npz in the state format (walked maps, star and flux refitted to the start images,
the other fields as given) and <out>.log, one line per trial step:

    step 7 a 0.050: maps 212.4 nm RMS, invisible 141.3 nm, image change 23.1 (budget 113), window share 0.998,
        chi2/N 0.741 (start 0.733), CG 61 it, 4.2 s

  - maps: RMS of the walked maps over their supports (pupil and Lyot stop openings, together).
  - invisible: RMS of the part of the maps the frames do not see (removed a fraction at a time).
  - image change: sum over the frames' pixels of ((image - start image) / noise)^2, star and
    flux free; a step is accepted while it stays within the budget (default sqrt(2 N) for
    N pixels: well inside the noise of a chi^2 over N pixels).
  - window share: detector-window light over post-stop light, relative to the start.
  - chi2/N: noise-weighted chi^2 per pixel of the model against the data, flux fitted.

The walk stops after --steps accepted steps, or when even a step of 1/64 of
--step_fraction leaves the budget.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn


#### WALK (model-independent)
def jacobian_products(residual, theta):
    """Residual and Jacobian products of a residual function at one parameter vector.

    Parameters
    ----------
    residual : callable
        Map from a parameter vector, shape (n,), to a residual vector, shape (m,).
    theta : torch.Tensor
        Parameter vector, shape (n,).

    Returns
    -------
    r : torch.Tensor
        Residual at theta, shape (m,).
    jvp : callable
        v -> J v, with J the Jacobian of residual at theta.
    vjp : callable
        u -> J^T u.

    Notes
    -----
    J v is the vector-Jacobian product of the linear map u -> J^T u, so only reverse-mode
    derivatives of the residual are needed.
    """
    r, vjp_fn = torch.func.vjp(residual, theta)
    transpose = lambda u: vjp_fn(u)[0]
    jvp = lambda v: torch.func.vjp(transpose, torch.zeros_like(r))[1](v)[0].detach()
    return r.detach(), jvp, lambda u: transpose(u).detach()


def nuisance_projector(columns):
    """Orthogonal projector removing the span of nuisance columns.

    Parameters
    ----------
    columns : torch.Tensor or None
        Residual derivatives along the nuisance parameters, shape (m, k).

    Returns
    -------
    callable
        u -> u - Q Q^T u, with Q an orthonormal basis of the columns.
    """
    if columns is None or columns.shape[1] == 0:
        return lambda u: u
    q = torch.linalg.qr(columns)[0]
    return lambda u: u - q @ (q.T @ u)


def conjugate_gradient(apply_A, b, n_iter, tol, apply_Minv=None):
    """Solve A x = b for a symmetric positive-definite A by preconditioned conjugate gradients.

    Parameters
    ----------
    apply_A : callable
        x -> A x.
    b : torch.Tensor
        Right-hand side, shape (m,).
    n_iter : int
        Maximum number of iterations.
    tol : float
        Stop when ||b - A x|| <= tol ||b||.
    apply_Minv : callable, optional
        x -> M^-1 x for a symmetric positive-definite preconditioner M; None uses none.

    Returns
    -------
    x : torch.Tensor
        Solution, shape (m,).
    n : int
        Number of iterations.
    rel : float
        Final relative residual norm.
    """
    x = torch.zeros_like(b)
    b_norm = float(b.norm())
    if b_norm == 0:
        return x, 0, 0.
    apply_Minv = apply_Minv or (lambda v: v)
    r = b.clone()
    z = apply_Minv(r)
    d = z.clone()
    rz = r @ z
    for n in range(1, n_iter + 1):
        Ad = apply_A(d)
        alpha = rz / (d @ Ad)
        x = x + alpha * d
        r = r - alpha * Ad
        if float(r.norm()) <= tol * b_norm:
            break
        z = apply_Minv(r)
        rz_new = r @ z
        d = z + (rz_new / rz) * d
        rz = rz_new
    return x, n, float(r.norm()) / b_norm


def nystrom_preconditioner(apply_K, m, rank, delta, dtype=torch.float64, device=None):
    """Randomized Nystrom preconditioner for K + delta I, with K symmetric positive semi-definite.

    Parameters
    ----------
    apply_K : callable
        x -> K x, for x of shape (m,).
    m : int
        Size of K.
    rank : int
        Number of products with K, the rank of the approximation.
    delta : float
        Shift of the system the preconditioner is for.
    dtype : torch.dtype, optional
        Real dtype of the vectors.
    device : torch.device or str, optional
        Device of the vectors.

    Returns
    -------
    callable
        x -> P^-1 x, with P^-1 = (lam_k + delta) U (Lam + delta I)^-1 U^T + (I - U U^T) for the
        rank-k approximation U Lam U^T of K.

    Notes
    -----
    Frangella, Tropp and Udell, SIAM J. Matrix Anal. Appl. 44, 718 (2023).
    """
    omega = torch.linalg.qr(torch.randn(m, rank, dtype=dtype, device=device))[0]
    Y = torch.stack([apply_K(omega[:, j]) for j in range(rank)], 1)
    nu = torch.finfo(dtype).eps * m**0.5 * float(Y.norm())
    Y = Y + nu * omega
    L = torch.linalg.cholesky(omega.T @ Y)
    B = torch.linalg.solve_triangular(L, Y.T, upper=False).T
    U, sv, _ = torch.linalg.svd(B, full_matrices=False)
    lam = torch.clamp(sv**2 - nu, min=0.)
    scale = (lam[-1] + delta) / (lam + delta)

    def apply(x):
        c = U.T @ x
        return U @ (scale * c) + (x - U @ c)
    return apply


def minimum_norm_change(jvp, vjp, project, support, b, delta, n_iter=1000, tol=1e-3, apply_Minv=None):
    """Smallest supported parameter change whose first-order residual change is b.

    Parameters
    ----------
    jvp, vjp : callable
        Jacobian products at the current parameters (see jacobian_products).
    project : callable
        Projector removing the nuisance directions from a residual vector.
    support : torch.Tensor
        1 on the parameters that may change, 0 elsewhere, shape (n,).
    b : torch.Tensor
        Residual change, shape (m,).
    delta : float
        Tikhonov parameter; directions with squared singular value below it count as invisible.
    n_iter : int, optional
        Maximum number of conjugate-gradient iterations.
    tol : float, optional
        Relative tolerance of the conjugate-gradient solve.
    apply_Minv : callable, optional
        Preconditioner of W J S J^T W + delta I (see nystrom_preconditioner).

    Returns
    -------
    dtheta : torch.Tensor
        Parameter change, shape (n,).
    n : int
        Number of conjugate-gradient iterations.
    rel : float
        Final relative residual of the solve.

    Notes
    -----
    dtheta = S J^T W y with (W J S J^T W + delta I) y = W b, where S is the support and W the projector.
    """
    apply_A = lambda y: project(jvp(support * vjp(project(y)))) + delta * y
    y, n, rel = conjugate_gradient(apply_A, project(b), n_iter, tol, apply_Minv)
    return support * vjp(project(y)), n, rel


def null_space_walk(residual, nuisance_columns, theta, support, steps, delta, budget, step_fraction=0.1,
                    retractions=4, retraction_delta=None, n_iter=1000, tol=1e-3, rank=0, report=None):
    """Remove the invisible part of the supported parameters while the residual stays near zero.

    Parameters
    ----------
    residual : callable
        Map from parameters, shape (n,), to the residual from the start state, shape (m,);
        zero at the start.
    nuisance_columns : callable or None
        Map from parameters to residual derivatives along the nuisance parameters, shape (m, k).
    theta : torch.Tensor
        Start parameters, shape (n,).
    support : torch.Tensor
        1 on the parameters that are walked, 0 elsewhere, shape (n,).
    steps : int
        Number of accepted steps.
    delta : float
        Tikhonov parameter that separates the invisible part (see minimum_norm_change).
    budget : float
        Largest squared norm of the projected residual of an accepted state.
    step_fraction : float, optional
        Largest fraction of the invisible part removed in one step.
    retractions : int, optional
        Gauss-Newton steps that pull each trial back to zero residual.
    retraction_delta : float, optional
        Tikhonov parameter of the retraction solves; None uses delta.
    n_iter, tol : optional
        Conjugate-gradient iteration limit and relative tolerance.
    rank : int, optional
        Rank of a Nystrom preconditioner of the invisible-part solves, built at the start and
        rebuilt at the current accepted parameters when a solve there misses the tolerance; 0
        uses none.
    report : callable, optional
        Called after each trial with (accepted, info dict).

    Returns
    -------
    theta : torch.Tensor
        Walked parameters, shape (n,).
    history : list of dict
        Per accepted step: step fraction, norm of the supported parameters, norm of their invisible
        part, squared projected residual after retraction, the largest number of conjugate-gradient
        iterations and relative residual, wall time [s].

    Notes
    -----
    Each step removes a fraction of the invisible part at the current parameters, then pulls the
    residual (always measured from the start state) back towards zero with the Jacobian at the
    trial parameters, so residual left by earlier steps is corrected too. What the retraction
    cannot correct are directions below delta; removing an invisible part of norm v costs at most
    delta v^2, so delta <= budget / v^2 keeps the walk within its budget.
    """
    retraction_delta = delta if retraction_delta is None else retraction_delta

    def linearise(t):
        r, jvp, vjp = jacobian_products(residual, t)
        project = nuisance_projector(None if nuisance_columns is None else nuisance_columns(t))
        return project(r), jvp, vjp, project

    def cost_at(t, project):
        with torch.no_grad():
            r = project(residual(t))
        return float(r @ r)

    preconditioner = [None, 0]

    def build(jvp, vjp, project, m, dtype, device):
        apply_K = lambda y: project(jvp(support * vjp(project(y))))
        preconditioner[0] = nystrom_preconditioner(apply_K, m, rank, delta, dtype, device)
        preconditioner[1] += 1

    history, a = [], step_fraction
    r, jvp, vjp, project = linearise(theta)
    if rank:
        build(jvp, vjp, project, len(r), r.dtype, r.device)
    while len(history) < steps and a > step_fraction / 64:
        t0, cg, cg_rel = time.time(), [], []
        s = support * theta
        visible, n, rel = minimum_norm_change(jvp, vjp, project, support, jvp(s), delta, n_iter, tol, preconditioner[0])
        if rank and rel > tol:
            build(jvp, vjp, project, len(r), r.dtype, r.device)
            visible, n, rel = minimum_norm_change(jvp, vjp, project, support, jvp(s), delta, n_iter, tol, preconditioner[0])
        cg.append(n)
        cg_rel.append(rel)
        invisible = s - visible
        trial = theta - a * invisible
        r_t, jvp_t, vjp_t, project_t = linearise(trial)
        for _ in range(retractions):
            dtheta, n, rel = minimum_norm_change(jvp_t, vjp_t, project_t, support, r_t, retraction_delta, n_iter, tol,
                                                 preconditioner[0])
            cg.append(n)
            cg_rel.append(rel)
            current = float(r_t @ r_t)
            for c in (1., 0.5, 0.25, 0.1):
                if cost_at(trial - c * dtheta, project_t) < current:
                    break
            else:
                break
            trial = trial - c * dtheta
            r_t, jvp_t, vjp_t, project_t = linearise(trial)
        cost = float(r_t @ r_t)
        accept = cost <= budget
        info = dict(step_fraction=a, norm=float((support * trial).norm()), invisible=float(invisible.norm()), cost=cost,
                    cg=max(cg), cg_residual=max(cg_rel), time=time.time() - t0)
        if report is not None:
            report(accept, trial, info)
        if accept:
            theta, r, jvp, vjp, project = trial, r_t, jvp_t, vjp_t, project_t
            history.append(info)
            a = min(step_fraction, 1.5 * a)
        else:
            a = a / 2
    return theta, history


#### PYFLIP MODEL
WF_NPIX, DIAMETER = 1024, 6.603464
PUPIL = 'instrument.optical_system.JWSTPrimary.wfe_offsets.grid'
LYOT = 'instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid'
STAR = 'wavefront.angles_offset'


def read_map(path):
    """A fitted OPD map [m] from a .npy file, leading axes of length 1 dropped."""
    m = np.load(path).astype(np.float64)
    return m.reshape(m.shape[-2:])


def state_from_args(args):
    """Fitted state from --state, or from PyFLIP map files and the options given."""
    if args.state:
        z = np.load(args.state)
        state = {k: z[k] for k in z.files if k != 'options'}
        state['options'] = json.loads(str(z['options']))
        return state
    options = dict(model_dir=args.model_dir, data_file=args.data, px_mask=args.px_mask, num_det_px=args.num_det_px, oversample=2,
                   fpm_raster=args.fpm_raster, um_per_arcsec=args.um_per_arcsec, band_limited=not args.full_raster,
                   square_cells=args.square_cells, mount=not args.no_mount, lyot_file=args.lyot_file, fpm_focal_length=131.4,
                   substrate=not args.no_substrate)
    star = None if args.star_mas is None else np.radians(np.array([float(v) for v in args.star_mas.split(',')]) / 3.6e6)
    return dict(pupil=read_map(args.pupil_map), lyot=read_map(args.lyot_map),
                flat=None if args.flat is None else np.squeeze(np.load(args.flat)), star=star, flux=None,
                registration=None if args.registration is None else np.array([float(v) for v in args.registration.split(',')]),
                blur=args.blur, lyot_shift=np.array([float(v) for v in args.lyot_shift.split(',')]), clocking=args.clocking,
                fpm_dz=1e-3 * args.fpm_dz, options=options)


def build_model(state, device):
    """PyFLIP instrument and wavefront for a fitted state (maps and star are set by the walk)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'refactoring'))
    from wavefronts import BroadbandWavefront
    from model_building import assemble_JWST_NIRCam_coron
    from dl_utils import load_model_data
    o = state['options']
    _, _, _, _, wlen_weights, _ = load_model_data(o['model_dir'], substrate=o.get('substrate', True))
    babinet = o.get('fpm_raster') is not None or not o.get('mount', True)
    instrument = assemble_JWST_NIRCam_coron(o['model_dir'], o['num_det_px'], DEVICE=device, oversample=o.get('oversample', 2),
                                            babinet=babinet, mount=o.get('mount', True), fpm_raster=o.get('fpm_raster'),
                                            um_per_arcsec=o.get('um_per_arcsec') or 593.336, band_limited=o.get('band_limited', True),
                                            square_cells=o.get('square_cells', False),
                                            fpm_axial_offset=float(state.get('fpm_dz', 0.)) or None,
                                            fpm_focal_length=o.get('fpm_focal_length', 131.4))
    system = instrument.optical_system
    lyot = system.NIRCamLyotstop
    if o.get('lyot_file'):
        lyot.amplitude = torch.FloatTensor(np.load(o['lyot_file'])).to(device)[None]
    detector = system.NIRCamdetector
    if state.get('registration') is not None:
        detector.subpixel_shift.shift_x, detector.subpixel_shift.shift_y = (float(v) for v in state['registration'])
    with torch.no_grad():
        lyot.pupil_shifts.x_shift.fill_(float(state['lyot_shift'][0]))
        lyot.pupil_shifts.y_shift.fill_(float(state['lyot_shift'][1]))
        lyot.pupil_shifts.rotation.fill_(float(state['clocking']))
        if state.get('blur') is not None:
            detector.charge_diffusion.log_sigma.fill_(np.log(float(state['blur']) * o.get('oversample', 2)))
        if state.get('flat') is not None:
            detector.detector_flat_field.grid.copy_(torch.as_tensor(state['flat']).reshape(detector.detector_flat_field.grid.shape))
    wavefront = BroadbandWavefront(WF_NPIX, DIAMETER, wlen_weights, torch.ones(1, dtype=torch.float64),
                                   angles_offset=torch.zeros(2, dtype=torch.float64), DEVICE=device).to(device)
    for parameter in list(instrument.parameters()) + list(wavefront.parameters()):
        parameter.requires_grad_(False)
    module = nn.ModuleDict({'instrument': instrument, 'wavefront': wavefront})
    module.forward = lambda: instrument(wavefront)
    return module


def load_frame(path, options, device):
    """Data, noise and the used pixels of one frame, flattened to those pixels."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'refactoring'))
    from dl_utils import load_and_crop_data
    data, noise = (np.squeeze(a).astype(np.float64) for a in load_and_crop_data(path, num_det_px=options['num_det_px'], coron_type='MASK335R'))
    good = np.isfinite(data) & np.isfinite(noise) & (noise > 0)
    if options.get('px_mask'):
        good &= np.squeeze(np.load(options['px_mask'])) > 0
    pixels = torch.as_tensor(np.flatnonzero(good), device=device)
    return (torch.as_tensor(data, device=device).reshape(-1)[pixels], torch.as_tensor(noise, device=device).reshape(-1)[pixels], pixels)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--state', help='state saved by flip_stages_fixed_geometry.py (fitted_*.npz)')
    parser.add_argument('--pupil_map', help='fitted JWSTPrimary OPD offsets (.npy, m)')
    parser.add_argument('--lyot_map', help='fitted NIRCamLyotstop OPD offsets (.npy, m)')
    parser.add_argument('--model_dir', help='PyFLIP model directory the maps were fitted with')
    parser.add_argument('--data', help='data frame the maps were fitted to (calints FITS or coadded .npy)')
    parser.add_argument('--px_mask', help='pixel mask (.npy, 1 = used)')
    parser.add_argument('--num_det_px', type=int, default=80, help='detector window [pixels]')
    parser.add_argument('--star_mas', help='fitted star offset x,y [mas]; default: fitted to the data')
    parser.add_argument('--flat', help='fitted detector flat field (.npy)')
    parser.add_argument('--registration', help='detector registration x,y [pixels]; default: the model directory\'s')
    parser.add_argument('--blur', type=float, help='detector blur sigma [pixels]; default: PyFLIP\'s')
    parser.add_argument('--lyot_shift', default='0,0', help='Lyot stop shift x,y [pupil samples]')
    parser.add_argument('--clocking', type=float, default=0., help='Lyot stop rotation [deg]')
    parser.add_argument('--fpm_dz', type=float, default=0., help='focal plane mask axial offset [mm, primary equivalent]')
    parser.add_argument('--lyot_file', help='Lyot stop transmission replacing the model directory\'s (.npy)')
    parser.add_argument('--fpm_raster', help='mask raster replacing the spot (see load_raster_region)')
    parser.add_argument('--um_per_arcsec', type=float, default=593.336, help='raster scale on the sky [um/arcsec]')
    parser.add_argument('--full_raster', action='store_true', help='every raster cell instead of the band-limited raster')
    parser.add_argument('--square_cells', action='store_true', help='square raster cells (band-limited raster)')
    parser.add_argument('--no_mount', action='store_true', help='drop the mask mount regions')
    parser.add_argument('--no_substrate', action='store_true', help='band weights without the mask substrate transmission')
    parser.add_argument('--frames', default='', help='further data frames whose images are kept (comma separated)')
    parser.add_argument('--steps', type=int, default=30, help='accepted steps')
    parser.add_argument('--step_fraction', type=float, default=0.1, help='largest fraction of the invisible part removed per step')
    parser.add_argument('--budget', type=float, help='largest image change, sum of ((image - start) / noise)^2; default sqrt(2 N)')
    parser.add_argument('--delta', type=float, help='image change per (nm RMS)^2 below which a map direction is invisible; '
                        'default budget / (4 x the squared start map RMS), so removing all of it stays within the budget')
    parser.add_argument('--share_weight', type=float, default=300., help='weight of the window-share rows; 0 does not keep the share')
    parser.add_argument('--rank', type=int, default=800, help='rank of the Nystrom preconditioner; 0: none')
    parser.add_argument('--n_iter', type=int, default=300, help='conjugate-gradient iterations per solve')
    parser.add_argument('--out', help='output name without extension; default: <state or pupil map>_walked')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if not args.state and not (args.pupil_map and args.lyot_map and args.model_dir and args.data):
        parser.error('give --state, or --pupil_map, --lyot_map, --model_dir and --data')
    device = args.device
    out = args.out or os.path.splitext(args.state or args.pupil_map)[0] + '_walked'
    log = open(out + '.log', 'w')

    def say(line):
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()

    state = state_from_args(args)
    options = state['options']
    module = build_model(state, device)
    lyot_optic = module['instrument'].optical_system.NIRCamLyotstop
    post_stop = {}
    lyot_optic.register_forward_hook(lambda m, i, output: post_stop.update(field=output))
    wl_weights = module['wavefront'].wl_weights
    files = [options['data_file']] + [f for f in args.frames.split(',') if f]
    frames = [load_frame(f, options, device) for f in files]

    shape = module['instrument'].optical_system.JWSTPrimary.wfe_offsets.grid.shape
    n_map = int(np.prod(shape))
    supports = (module['instrument'].optical_system.JWSTPrimary.amplitude.reshape(-1) > 0,
                lyot_optic.transmission().detach().reshape(-1) > 1e-3)
    support = torch.cat(supports).double()
    unit = 1e-9 * float(support.sum())**0.5                    # theta of norm 1 over the supports: maps of 1 nm RMS [m]
    theta0 = torch.as_tensor(np.concatenate([np.asarray(state['pupil']).reshape(-1), np.asarray(state['lyot']).reshape(-1)]),
                             dtype=torch.float64, device=device) / unit

    def render(theta, star):
        """Model image on the used pixels of every frame's window, and the window share of the post-stop light."""
        params = {PUPIL: (unit * theta[:n_map]).reshape(shape), LYOT: (unit * theta[n_map:]).reshape(shape), STAR: star}
        image = torch.func.functional_call(module, params, ()).reshape(-1)
        energy = (post_stop['field'].abs()**2 * wl_weights.view(-1, 1, 1)).sum()
        return image, image.sum() / energy

    def fit_star_flux(theta, star, k, steps=6, h=np.radians(5e-4 / 3600)):
        """Star (Gauss-Newton) and least-squares flux of frame k against its data."""
        data, noise, pixels = frames[k]
        def r_of(s):
            with torch.no_grad():
                image = render(theta, s)[0][pixels]
            flux = (image * data / noise**2).sum() / (image**2 / noise**2).sum()
            return (flux * image - data) / noise, flux
        for _ in range(steps):
            r = r_of(star)[0]
            J = torch.stack([(r_of(star + h * e)[0] - r) / h for e in torch.eye(2, dtype=torch.float64, device=device)], 1)
            star = star + torch.linalg.lstsq(J, -r[:, None]).solution[:, 0]
        r, flux = r_of(star)
        return star, float(flux), float(r @ r) / len(r)

    stars, fluxes = [], []
    for k in range(len(frames)):
        star = torch.as_tensor(state['star'], dtype=torch.float64, device=device) if (k == 0 and state.get('star') is not None) \
            else (stars[0].clone() if stars else torch.zeros(2, dtype=torch.float64, device=device))
        if k > 0 or state.get('star') is None:
            star = fit_star_flux(theta0, star, k)[0]
        stars.append(star)
        data, noise, pixels = frames[k]
        with torch.no_grad():
            image = render(theta0, star)[0][pixels]
        fluxes.append(float((image * data / noise**2).sum() / (image**2 / noise**2).sum()))
    with torch.no_grad():
        start = [render(theta0, s) for s in stars]
    images0 = [im[f[2]] for (im, _), f in zip(start, frames)]
    shares0 = [float(sh) for _, sh in start]
    n_pixels = sum(len(f[2]) for f in frames)
    budget = args.budget if args.budget is not None else (2 * n_pixels)**0.5

    def residual_with(theta, star_list):
        rows = []
        for k, ((data, noise, pixels), star) in enumerate(zip(frames, star_list)):
            image, share = render(theta, star)
            rows.append((fluxes[k] * image[pixels] - fluxes[k] * images0[k]) / noise)
            if args.share_weight:
                rows.append(args.share_weight * (share / shares0[k] - 1)[None])
        return torch.cat(rows)

    residual = lambda theta: residual_with(theta, stars)

    def nuisance_columns(theta):
        columns = []
        for k in range(len(frames)):
            _, jvp, _ = jacobian_products(lambda s: residual_with(theta, stars[:k] + [s] + stars[k + 1:]), stars[k])
            columns += [jvp(e) for e in torch.eye(2, dtype=torch.float64, device=device)]
            flux_column, offset = torch.zeros_like(columns[-1]), 0
            for j, (data, noise, pixels) in enumerate(frames):
                n = len(pixels)
                if j == k:
                    with torch.no_grad():
                        flux_column[offset:offset + n] = fluxes[k] * render(theta, stars[k])[0][pixels] / noise
                offset += n + (1 if args.share_weight else 0)
            columns.append(flux_column)
        return torch.stack(columns, 1)

    def chi2(theta):
        values = []
        for k, (data, noise, pixels) in enumerate(frames):
            with torch.no_grad():
                image = render(theta, stars[k])[0][pixels]
            flux = (image * data / noise**2).sum() / (image**2 / noise**2).sum()
            values.append(float((((flux * image - data) / noise)**2).mean()))
        return values

    def share_ratio(theta):
        with torch.no_grad():
            return [float(render(theta, s)[1]) / s0 for s, s0 in zip(stars, shares0)]

    def rms(theta):
        return float((support * theta).norm())

    chi2_0 = chi2(theta0)
    delta = args.delta if args.delta is not None else budget / (4 * float((support * theta0).norm())**2)
    say('frames: %s | pixels %d | budget %.1f | delta %.3g | share weight %g' % (', '.join(os.path.basename(f) for f in files), n_pixels,
                                                                              budget, delta, args.share_weight))
    say('start: maps %.1f nm RMS | chi2/N %s' % (rms(theta0), ' '.join('%.4f' % c for c in chi2_0)))

    def report(accepted, theta, info):
        say('%s a %.3f: maps %.1f nm RMS, invisible %.1f nm, image change %.1f (budget %.0f), window share %s, chi2/N %s (start %s), '
            'CG %d it (residual %.1e), %.1f s' % ('step %d' % (len(history_box) + 1) if accepted else 'rejected', info['step_fraction'],
                                                  info['norm'], info['invisible'], info['cost'], budget,
                                                  ' '.join('%.4f' % v for v in share_ratio(theta)), ' '.join('%.4f' % c for c in chi2(theta)),
                                                  ' '.join('%.4f' % c for c in chi2_0), info['cg'], info['cg_residual'], info['time']))
        if accepted:
            history_box.append(info)

    history_box = []
    t0 = time.time()
    theta, history = null_space_walk(residual, nuisance_columns, theta0, support, args.steps, delta, budget, args.step_fraction,
                                     n_iter=args.n_iter, rank=args.rank, report=report)
    walked_stars, walked_fluxes = [], []
    for k in range(len(frames)):
        data, noise, pixels = frames[k]
        star, h = stars[k], np.radians(5e-4 / 3600)
        def r_of(s):
            with torch.no_grad():
                image = render(theta, s)[0][pixels]
            flux = (image * images0[k]).sum() / (image**2).sum()
            return (flux * image - images0[k]) / noise, flux
        for _ in range(6):
            r = r_of(star)[0]
            J = torch.stack([(r_of(star + h * e)[0] - r) / h for e in torch.eye(2, dtype=torch.float64, device=device)], 1)
            star = star + torch.linalg.lstsq(J, -r[:, None]).solution[:, 0]
        walked_stars.append(star)
        walked_fluxes.append(fluxes[k] * float(r_of(star)[1]))
    stars = walked_stars
    say('walked: %d steps in %.0f s | maps %.1f -> %.1f nm RMS | chi2/N %s (start %s) | window share %s' % (
        len(history), time.time() - t0, rms(theta0), rms(theta), ' '.join('%.4f' % c for c in chi2(theta)),
        ' '.join('%.4f' % c for c in chi2_0), ' '.join('%.4f' % v for v in share_ratio(theta))))

    walked = {k: v for k, v in state.items() if k != 'options' and v is not None}
    walked['pupil'] = (unit * theta[:n_map]).reshape(shape[-2:]).cpu().numpy()
    walked['lyot'] = (unit * theta[n_map:]).reshape(shape[-2:]).cpu().numpy()
    walked['star'], walked['flux'] = walked_stars[0].cpu().numpy(), walked_fluxes[0]
    np.savez(out + '.npz', options=json.dumps(options), walk_history=json.dumps(history),
             walk_options=json.dumps(vars(args)), **walked)
    say('saved %s.npz and %s.log' % (out, out))


if __name__ == '__main__':
    main()
