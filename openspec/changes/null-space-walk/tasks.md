# Tasks

- [x] 1. Walk functions: Jacobian products, nuisance projector, preconditioned conjugate gradients,
      minimum-norm change, walk; tests on a linear model and a small optical system.
- [x] 2. `flip_stages_fixed_geometry.py`: save the fitted state (maps, flat, star, flux, geometry, options).
- [x] 3. `experiments/null_space_walk.py`: one script that rebuilds the model from a saved state or
      PyFLIP `.npy` maps, walks the OPD maps keeping images and window share, saves the result.
- [x] 4. Validate on fractal: a short one-wavelength fit, then a walk; report map RMS, image change,
      window share, chi^2/N on the training frame, CG iterations and time per step.
