# Design

## Quantities

- theta: the maps flattened into one vector (pupil OPD, then Lyot-plane OPD), scaled so that a unit
  vector is a 1 nm RMS map over the supports. A support mask S keeps each map on its own aperture
  (primary transmission > 0, Lyot stop > 0).
- r(theta): per frame, the whitened image change from the start images, (f I(theta) - f0 I0) / sigma
  on the unmasked pixels, then w (s(theta) / s0 - 1), with s the window share: detector-window light
  over post-stop light.
- J: the Jacobian of r at the current theta, used only through products J v and J^T u, both from
  reverse mode (J v as the derivative of a vjp with respect to its cotangent).
- N: whitened image derivatives along the nuisances (star position, flux, per frame). W = I - Q Q^T
  with Q an orthonormal basis of N removes them, so image changes a nuisance refit absorbs are free.

## Minimum-norm change

For an image change b, the smallest change of theta with W J S dtheta = W b is

    dtheta = S J^T W y,   (W J S J^T W + delta I) y = W b,

solved by conjugate gradients on image-sized vectors, preconditioned by a rank-k randomized Nystrom
approximation of W J S J^T W. delta sets which directions count as visible: those whose image
change per unit theta is well above sqrt(delta). Default delta = budget / (4 |S theta0|^2).

## Step

    v         = S theta - dtheta(J S theta)            invisible part of the map
    trial     = theta - a v
    retract:  trial <- trial - c dtheta(W r(trial)),   c in (1, 1/2, 1/4, 1/10), while |W r| decreases,
              J at the trial point
    accept if |W r(trial)|^2 <= budget (then a <- min(a0, 1.5 a)), else a <- a / 2

Default budget sqrt(2 M) for M pixels. The preconditioner is rebuilt only at accepted states, and only
when a solve misses its tolerance.

## Choices

- float64 for theta and the linear algebra; the model is evaluated in complex128.
- One file, PyFLIP modules imported inside the model functions, so the walk and its tests run
  without stpsf or data.
