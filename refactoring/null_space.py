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
    """
    r, vjp_fn = torch.func.vjp(residual, theta)
    jvp = lambda v: torch.func.jvp(residual, (theta,), (v,))[1]
    return r.detach(), jvp, lambda u: vjp_fn(u)[0]


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


def conjugate_gradient(apply_A, b, n_iter, tol):
    """Solve A x = b for a symmetric positive-definite A by conjugate gradients.

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
    r = b.clone()
    d = r.clone()
    rr = r @ r
    for n in range(1, n_iter + 1):
        Ad = apply_A(d)
        alpha = rr / (d @ Ad)
        x = x + alpha * d
        r = r - alpha * Ad
        rr_new = r @ r
        if float(rr_new.sqrt()) <= tol * b_norm:
            break
        d = r + (rr_new / rr) * d
        rr = rr_new
    return x, n, float(rr_new.sqrt()) / b_norm


def minimum_norm_change(jvp, vjp, project, support, b, delta, n_iter=300, tol=1e-4):
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
    y, n, rel = conjugate_gradient(apply_A, project(b), n_iter, tol)
    return support * vjp(project(y)), n, rel


def null_space_walk(residual, nuisance_columns, theta, support, steps, delta, budget, step_fraction=0.1,
                    retractions=4, n_iter=300, tol=1e-4, verbose=False):
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
    verbose : bool, optional
        Print one line per trial.

    Returns
    -------
    theta : torch.Tensor
        Walked parameters, shape (n,).
    history : list of dict
        Per accepted step: step fraction, norm of the supported parameters, norm of their invisible
        part before the step, squared projected residual before and after retraction, the largest
        number of conjugate-gradient iterations and the step's wall time [s].
    """
    def linearise(t):
        r, jvp, vjp = jacobian_products(residual, t)
        project = nuisance_projector(None if nuisance_columns is None else nuisance_columns(t))
        return project(r), jvp, vjp, project

    def cost_at(t, project):
        with torch.no_grad():
            r = project(residual(t))
        return float(r @ r)

    history, a, start = [], step_fraction, time.time()
    while len(history) < steps and a > step_fraction / 64:
        t0, cg = time.time(), []
        _, jvp, vjp, project = linearise(theta)
        s = support * theta
        visible, n, _ = minimum_norm_change(jvp, vjp, project, support, jvp(s), delta, n_iter, tol)
        cg.append(n)
        invisible = s - visible
        trial = theta - a * invisible
        r, jvp, vjp, project = linearise(trial)
        before = float(r @ r)
        for _ in range(retractions):
            dtheta, n, _ = minimum_norm_change(jvp, vjp, project, support, r, delta, n_iter, tol)
            cg.append(n)
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
            print('%s a %.3f: |theta| %.4g, invisible %.4g, |W r|^2 %.3g -> %.3g, CG max %d, %.1f s'
                  % ('step %d' % (len(history) + 1) if accept else 'rejected', a, float(s.norm()),
                     float(invisible.norm()), before, after, max(cg), time.time() - t0), flush=True)
        if accept:
            theta = trial
            history.append(dict(step_fraction=a, norm=float((support * theta).norm()), invisible=float(invisible.norm()),
                                cost_before=before, cost_after=after, cg_max=max(cg), time=time.time() - t0))
            a = min(step_fraction, 1.5 * a)
        else:
            a = a / 2
    if verbose:
        print('walk: %d steps, %.0f s' % (len(history), time.time() - start), flush=True)
    return theta, history
