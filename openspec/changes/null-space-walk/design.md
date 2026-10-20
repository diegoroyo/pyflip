# Design

## Quantities

- theta: the fitted maps flattened into one vector (pupil OPD offset, then NIRCam OPD offset).
  A support mask S keeps each map on its own aperture (primary transmission > 0, Lyot stop > 0).
- r(theta): whitened image change of the training frames from the start model's images,
  (I(theta) - I(theta0)) / sigma on the unmasked pixels.
- J: the Jacobian of r at the current theta, used only through products J v (`torch.func.jvp`)
  and J^T u (`torch.func.vjp`).
- N: whitened image derivatives along the nuisance parameters (star position, flux). W = I - Q Q^T
  with Q an orthonormal basis of N removes them, so image changes a nuisance refit absorbs are free.

## Minimum-norm change

For an image change b, the smallest change of theta with W J S dtheta = W b is

    dtheta = S J^T W y,   (W J S J^T W + delta I) y = W b,

solved by conjugate gradients on image-sized vectors. delta sets which directions count as visible:
those whose image change per unit theta is well above sqrt(delta).

## Step

    v         = S theta - dtheta(J S theta)            invisible part of the map
    trial     = theta - a v
    retract:  trial <- trial - c dtheta(W r(trial)),   c in (1, 1/2, 1/4, 1/10), while ||W r|| decreases
    accept if ||W r(trial)||^2 <= budget (then a <- min(a0, 1.5 a)), else a <- a / 2

J and W are re-evaluated at every state where they are used.

## Choices

- No preconditioner at first; CG iteration counts are reported so one can be added if needed.
- float64 for theta and the linear algebra; the driver evaluates the model in complex128.
- The driver scales theta so that a unit vector is a 1 nm RMS map over the supports; delta is then a
  chi^2 change per (nm RMS)^2.
