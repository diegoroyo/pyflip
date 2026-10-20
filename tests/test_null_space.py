"""Null-space walk on a linear model and on a small Fraunhofer imaging system."""
import torch

from null_space import null_space_walk, nuisance_projector, jacobian_products


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


def test_walk_shrinks_map_of_small_imaging_system_within_budget():
    """On a cropped Fraunhofer image the walk drops the phase the crop cannot see and keeps the image."""
    n, pad, half = 32, 64, 7
    y, x = torch.meshgrid(torch.arange(n, dtype=torch.float64) - (n - 1) / 2,
                          torch.arange(n, dtype=torch.float64) - (n - 1) / 2, indexing='ij')
    aperture = (torch.hypot(x, y) < 12).to(torch.float64)
    smooth = 0.3 * x / 12 + 0.2 * (x**2 - y**2) / 144
    ripple = 0.2 * torch.cos(2 * torch.pi * 12 * x / n)

    def image(phase):
        field = torch.zeros(pad, pad, dtype=torch.complex128)
        field[:n, :n] = aperture * torch.exp(1j * phase.reshape(n, n))
        intensity = torch.fft.fftshift(torch.fft.fft2(field)).abs()**2
        return intensity[pad // 2 - half:pad // 2 + half + 1, pad // 2 - half:pad // 2 + half + 1].reshape(-1)

    theta0 = ((smooth + ripple) * aperture).reshape(-1)
    image0 = image(theta0)
    sigma = 1e-3 * image0.max()
    residual = lambda t: (image(t) - image0) / sigma
    support = aperture.reshape(-1)

    theta, history = null_space_walk(residual, lambda t: image(t)[:, None] / sigma, theta0, support,
                                     steps=10, delta=1., budget=1., step_fraction=0.3)

    flux = (image(theta) @ image0) / (image(theta) @ image(theta))
    assert len(history) == 10
    assert history[-1]['cost_after'] <= 1.
    assert float(((flux * image(theta) - image0) / sigma).norm()**2) <= 1.
    assert float((support * theta).norm()) < 0.8 * float((support * theta0).norm())
