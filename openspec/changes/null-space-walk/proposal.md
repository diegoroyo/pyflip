# Proposal

## Why

A fitted pupil or Lyot-plane OPD map is not unique: many maps give the same detector images to first
order, and the optimiser leaves an arbitrary part of the map in directions the data cannot see. That
part inflates the maps and is not a mirror measurement. A null-space walk moves a fitted model along
its equivalence class, removing the invisible part while keeping its images fixed, so the maps that
remain are the ones the data require.

## What Changes

- `refactoring/null_space.py`: the walk as a few small model-independent functions on a flat
  parameter vector: Jacobian products by `torch.func`, a projector that removes nuisance directions,
  conjugate gradients, the minimum-norm parameter change for an image change, and the walk itself.
- `experiments/null_space_walk.py`: a driver that rebuilds the `PointPropagate` model of a finished
  `run_FLIP_displacement.py` fit, walks its pupil and NIRCam OPD offsets with star position and flux
  as nuisances, and saves the walked parameters in the same format.
- `run_FLIP_displacement.py`: also save the fitted model parameters and the run's options
  (`fitted_parameters.npz`), so a walk or an evaluation can start from a fit.
- Tests: a linear model, where one full step must reach the minimum-norm solution, and a small optical
  system, where the images must stay within the budget while the map norm falls.

## Impact

New files only, plus one save at the end of the runner. No change to any forward model or fit.
