"""Null-space walk on a linear model and on a small Fraunhofer imaging system."""
import os
import sys
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, cg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments'))
from null_space_walk import null_space_walk, nuisance_projector, jacobian_products, nystrom_preconditioner


def test_jacobian_products_match_full_jacobian():
    """J v and J^T u from reverse mode alone equal products with the full Jacobian."""
    torch.manual_seed(1)
    A = torch.randn(7, 5, dtype=torch.float64)
    residual = lambda t: torch.sin(A @ t) * torch.exp(0.3 * (A @ t))
    theta, v, u = torch.randn(5, dtype=torch.float64), torch.randn(5, dtype=torch.float64), torch.randn(7, dtype=torch.float64)
    J = torch.autograd.functional.jacobian(residual, theta)
    r, jvp, vjp = jacobian_products(residual, theta)
    torch.testing.assert_close(r, residual(theta), rtol=0, atol=0)
    torch.testing.assert_close(jvp(v), J @ v, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(vjp(u), J.T @ u, rtol=1e-12, atol=1e-12)


def test_nystrom_preconditioner_speeds_up_conjugate_gradients():
    """A Nystrom preconditioner covering the eigenvalues above delta cuts scipy's CG iterations, same solution."""
    torch.manual_seed(2)
    m, delta = 200, 1e-6
    Q = torch.linalg.qr(torch.randn(m, m, dtype=torch.float64))[0]
    K = Q @ torch.diag(10.0**torch.linspace(4, -10, m, dtype=torch.float64)) @ Q.T
    A = (K + delta * torch.eye(m, dtype=torch.float64)).numpy()
    b = torch.randn(m, dtype=torch.float64).numpy()
    apply_Minv = nystrom_preconditioner(lambda x: K @ x, m, 160, delta, torch.zeros(m, dtype=torch.float64))
    M = LinearOperator((m, m), matvec=lambda x: apply_Minv(torch.as_tensor(x)).numpy())
    iterations = {}
    for name, preconditioner in (('plain', None), ('nystrom', M)):
        count = [0]
        x, _ = cg(A, b, rtol=1e-8, maxiter=500, M=preconditioner, callback=lambda _: count.__setitem__(0, count[0] + 1))
        iterations[name] = count[0]
    assert iterations['nystrom'] < iterations['plain'] / 10
    np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-6, atol=1e-6 * np.linalg.norm(np.linalg.solve(A, b)))


def test_full_step_on_linear_model_reaches_minimum_norm_solution():
    """One full step of a linear model lands on the minimum-norm parameters with the same projected images."""
    torch.manual_seed(0)
    A = torch.randn(20, 50, dtype=torch.float64)
    theta0 = torch.randn(50, dtype=torch.float64)
    nuisance = torch.randn(20, 1, dtype=torch.float64)
    project = nuisance_projector(nuisance)
    W = torch.eye(20, dtype=torch.float64) - nuisance @ nuisance.T / (nuisance.T @ nuisance)
    expected = torch.linalg.pinv(W @ A) @ (W @ A @ theta0)

    theta, history = null_space_walk(lambda t: A @ (t - theta0), lambda t: nuisance, theta0,
                                     torch.ones(50, dtype=torch.float64), steps=1, delta=1e-12, budget=1e-12,
                                     step_fraction=1.)

    assert len(history) == 1
    torch.testing.assert_close(theta, expected, rtol=0, atol=1e-8)
    assert float(project(A @ (theta - theta0)).norm()) < 1e-8


def small_imaging_system():
    """A cropped Fraunhofer image of a phase map on a round aperture, its start map and support."""
    n, pad, half = 32, 64, 7
    y, x = torch.meshgrid(torch.arange(n, dtype=torch.float64) - (n - 1) / 2,
                          torch.arange(n, dtype=torch.float64) - (n - 1) / 2, indexing='ij')
    aperture = (torch.hypot(x, y) < 12).to(torch.float64)
    smooth = 0.3 * x / 12 + 0.2 * (x**2 - y**2) / 144
    ripple = 0.2 * torch.cos(2 * torch.pi * 12 * x / n)

    def image(phase):
        """Cropped image and the share of the light inside the crop."""
        field = torch.zeros(pad, pad, dtype=torch.complex128)
        field[:n, :n] = aperture * torch.exp(1j * phase.reshape(n, n))
        intensity = torch.fft.fftshift(torch.fft.fft2(field)).abs()**2
        crop = intensity[pad // 2 - half:pad // 2 + half + 1, pad // 2 - half:pad // 2 + half + 1].reshape(-1)
        return crop, crop.sum() / intensity.sum()

    return image, ((smooth + ripple) * aperture).reshape(-1), aperture.reshape(-1)


def test_walk_shrinks_map_of_small_imaging_system_within_budget():
    """On a cropped Fraunhofer image the walk drops the phase the crop cannot see and keeps the image."""
    image, theta0, support = small_imaging_system()
    image0 = image(theta0)[0]
    sigma = 1e-3 * image0.max()
    residual = lambda t: (image(t)[0] - image0) / sigma

    theta, history = null_space_walk(residual, lambda t: image(t)[0][:, None] / sigma, theta0, support,
                                     steps=10, delta=1., budget=1., step_fraction=0.3)

    flux = (image(theta)[0] @ image0) / (image(theta)[0] @ image(theta)[0])
    assert len(history) == 10
    assert history[-1]['cost'] <= 1.
    assert float(((flux * image(theta)[0] - image0) / sigma).norm()**2) <= 1.
    assert float((support * theta).norm()) < 0.8 * float((support * theta0).norm())


def test_walk_keeps_the_window_share_of_the_light():
    """With a share row in the residual the crop keeps its share of the light, and the walk still
    removes invisible content that does not move light (a piston); without the row the walk also
    removes the ripple that scatters light out of the crop, and the share changes."""
    image, theta0, support = small_imaging_system()
    theta0 = theta0 + support                                # a piston of 1 rad over the aperture
    image0, share0 = image(theta0)
    sigma = 1e-3 * image0.max()
    shares, norms = {}, {}
    for weight in (0., 300.):
        def residual(t):
            crop, share = image(t)
            rows = [(crop - image0) / sigma] + ([weight * (share / share0 - 1)[None]] if weight else [])
            return torch.cat(rows)
        nuisance = lambda t: torch.cat([image(t)[0] / sigma] + ([torch.zeros(1, dtype=torch.float64)] if weight else []))[:, None]
        theta, history = null_space_walk(residual, nuisance, theta0, support, steps=10, delta=1., budget=1., step_fraction=0.3)
        assert len(history) == 10
        shares[weight], norms[weight] = float(image(theta)[1] / share0), float((support * theta).norm())
    assert abs(shares[300.] - 1) < 1e-3
    assert abs(shares[0.] - 1) > 10 * abs(shares[300.] - 1)
    assert norms[300.] < 0.8 * float((support * theta0).norm())
