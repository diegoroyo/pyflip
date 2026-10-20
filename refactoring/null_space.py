"""Null-space walk: move fitted parameters along directions their images cannot see."""
import time
import torch


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
                    retractions=4, n_iter=1000, tol=1e-3, rank=0, verbose=False):
    """Remove the invisible part of the supported parameters while holding the residual near zero.

    Parameters
    ----------
    residual : callable
        Map from parameters, shape (n,), to the whitened residual from the start model's images,
        shape (m,); zero at the start.
    nuisance_columns : callable or None
        Map from parameters to residual derivatives along the nuisance parameters, shape (m, k).
    theta : torch.Tensor
        Start parameters, shape (n,).
    support : torch.Tensor
        1 on the parameters that are walked, 0 elsewhere, shape (n,).
    steps : int
        Number of accepted steps.
    delta : float
        Tikhonov parameter of the minimum-norm solves (see minimum_norm_change).
    budget : float
        Largest squared norm of the projected residual of an accepted state.
    step_fraction : float, optional
        Largest fraction of the invisible part removed in one step.
    retractions : int, optional
        Gauss-Newton retraction steps per trial.
    n_iter, tol : optional
        Conjugate-gradient iteration limit and relative tolerance.
    rank : int, optional
        Rank of a Nystrom preconditioner, built at the start parameters and rebuilt at the current ones
        whenever a solve misses the tolerance; 0 uses none.
    verbose : bool, optional
        Print one line per trial.

    Returns
    -------
    theta : torch.Tensor
        Walked parameters, shape (n,).
    history : list of dict
        Per accepted step: step fraction, norm of the supported parameters, norm of their invisible
        part before the step, squared projected residual before and after retraction, the largest
        number of conjugate-gradient iterations and relative residual, the number of preconditioner
        builds and the step's wall time [s].
    """
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

    def solve(jvp, vjp, project, b):
        dtheta, n, rel = minimum_norm_change(jvp, vjp, project, support, b, delta, n_iter, tol, preconditioner[0])
        if rank and rel > tol:
            build(jvp, vjp, project, len(b), b.dtype, b.device)
            dtheta, n_more, rel = minimum_norm_change(jvp, vjp, project, support, b, delta, n_iter, tol, preconditioner[0])
            n += n_more
        return dtheta, n, rel

    history, a, start = [], step_fraction, time.time()
    if rank:
        r, jvp, vjp, project = linearise(theta)
        build(jvp, vjp, project, len(r), r.dtype, r.device)
    while len(history) < steps and a > step_fraction / 64:
        t0, cg, cg_rel, builds = time.time(), [], [], preconditioner[1]
        _, jvp, vjp, project = linearise(theta)
        s = support * theta
        visible, n, rel = solve(jvp, vjp, project, jvp(s))
        cg.append(n)
        cg_rel.append(rel)
        invisible = s - visible
        trial = theta - a * invisible
        r, jvp, vjp, project = linearise(trial)
        before = float(r @ r)
        for _ in range(retractions):
            dtheta, n, rel = solve(jvp, vjp, project, r)
            cg.append(n)
            cg_rel.append(rel)
            for c in (1., 0.5, 0.25, 0.1):
                if cost_at(trial - c * dtheta, project) < float(r @ r):
                    break
            else:
                break
            trial = trial - c * dtheta
            r, jvp, vjp, project = linearise(trial)
        after = float(r @ r)
        accept = after <= budget
        if verbose:
            print('%s a %.3f: |theta| %.4g, invisible %.4g, |W r|^2 %.3g -> %.3g, CG max %d it, residual %.1e, '
                  '%d preconditioner builds, %.1f s'
                  % ('step %d' % (len(history) + 1) if accept else 'rejected', a, float(s.norm()),
                     float(invisible.norm()), before, after, max(cg), max(cg_rel), preconditioner[1] - builds,
                     time.time() - t0), flush=True)
        if accept:
            theta = trial
            history.append(dict(step_fraction=a, norm=float((support * theta).norm()), invisible=float(invisible.norm()),
                                cost_before=before, cost_after=after, cg_max=max(cg), cg_residual=max(cg_rel),
                                builds=preconditioner[1] - builds, time=time.time() - t0))
            a = min(step_fraction, 1.5 * a)
        else:
            a = a / 2
    if verbose:
        print('walk: %d steps, %d preconditioner builds, %.0f s' % (len(history), preconditioner[1], time.time() - start),
              flush=True)
    return theta, history
