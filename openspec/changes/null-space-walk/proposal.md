# Proposal

## Why

A fitted pupil or Lyot-plane OPD map is not unique: many maps give the same detector images to first
order, and the optimiser leaves an arbitrary part of the map in directions the data cannot see. That
part inflates the maps and is not a mirror measurement. A null-space walk moves a fitted model along
its equivalence class, removing the invisible part while keeping its images fixed, so the maps that
remain are the ones the data require.

## What Changes

- `experiments/null_space_walk.py`: one self-contained script that can be sent to someone with their
  own fitted maps. It holds
  - the walk as small model-independent functions on a flat parameter vector: Jacobian products from
    reverse mode, a projector that removes nuisance directions, preconditioned conjugate gradients
    (randomized Nystrom), the minimum-norm parameter change for an image change, and the walk;
  - the model side: read a saved fit state or PyFLIP's `.npy` maps, rebuild the
    `assemble_JWST_NIRCam_coron` model with the fit's geometry, and walk the pupil and Lyot-plane
    OPD maps with per-frame star position and flux as nuisances, keeping each frame's share of the
    post-stop light inside the detector window;
  - a command line, a log line per trial step and an output in the saved-state format.
- `experiments/flip_stages_fixed_geometry.py`: save the fitted state (maps, flat, star, flux,
  geometry, options) after the reference stages and at the end, so a walk can start from it.
- Tests: Jacobian products against the full Jacobian, the preconditioner against plain CG, a linear
  model where one full step reaches the minimum-norm solution, and a small Fraunhofer system where
  the images stay within the budget, the map norm falls, and the window share is kept.

## Impact

One new script, one state save in the staged fit. No change to any forward model or fit.
