# AGENTS.md

## Project Scope
EMerge_Optimizer is a Python toolkit for building and optimizing IFA/MIFA antennas using the EMerge electromagnetic solver backend.

Primary workflows:
- Build and simulate a single antenna geometry.
- Build and simulate multi-stub (multi-frequency) geometries with alias-prefixed parameters.
- Run global optimization (differential evolution) and local optimization (Powell/Nelder-Mead) against S11 objectives.

## Tech Stack
- Python
- `emerge` solver API (`CUDSS` preferred, `PARDISO` fallback)
- `numpy`, `scipy`
- `multiprocessing` with `spawn`

## Key Files
- `ifalib2.py`: Core API for geometry construction, parameter normalization/link resolution, simulation, and post-processing helpers.
- `ifa_validation.py`: Geometry and mesh constraint validation used to reject infeasible candidates.
- `optimize_lib.py`: Objective functions, isolation worker process, global and local optimizers, logging, and bounds utilities.
- `demo_1_parametric_mifa_antenna.py`: Single-frequency build/sim/plot example.
- `demo_2_parametric_multifrequency_mifa2.py`: Multi-alias (`p.`, `p2.`) build/sim example.
- `demo_optimize_*` and `demo_parametric_*optimizer*`: Optimization entry scripts.

## Canonical APIs
- `build_mifa(params_any, ...) -> (model, S11, freq_dense, ff1, ff2, ff3d)`
- `normalize_params_sequence(params_any) -> List[AntennaParams]`
- `denormalize_params_sequence_flat(seq) -> Dict[str, Any]`
- `resolve_linked_params(raw_list)`
- `validate_ifa_params(params_any) -> (errors, warnings, derived)`
- `global_optimizer(...)`
- `local_minimize_ifa(...)`

## Parameter Conventions
Single-antenna mode:
- Use plain keys like `ifa_l`, `f0`, `f1`, `f2`, `freq_points`.

Multi-antenna mode:
- Use alias-prefixed flat keys like `p.ifa_l`, `p2.ifa_l`, etc.
- `p` is the base profile; `p2+` inherit unspecified fields from `p` after normalization.
- Expression links are supported, for example `${p.ifa_h} - 0.003`.

Frequency modes:
- Triplet mode: `f1/f0/f2` with `freq_points`.
- Sweep mode: `sweep_freqs` (optional `sweep_weights`).

Units:
- Geometry is in meters.
- `mm = 1e-3` is used in scripts for readability.

## Optimization Notes
- Global optimization uses `scipy.optimize.differential_evolution`.
- Each simulation eval is run in an isolated child process (`spawn`) to survive native solver crashes.
- Invalid designs are penalized via validator checks.
- Objectives support:
  - Sweep-frequency weighted objective.
  - Bandwidth-aware objective around target RL threshold.
  - Single-point fallback objective at `f0`.

Logging/output:
- Optimizer writes best candidates to `best_params_logs/` in `optimize_lib.py`.
- Some demos write to `best_params_log/` (singular). Keep this inconsistency in mind when looking for artifacts.

## Runtime Characteristics
- Simulations are computationally heavy.
- Prefer `CUDSS` solver when available.
- Keep demo `maxiter/popsize` conservative when testing changes.

## Safe Change Guidelines
- Preserve both plain-key and alias-prefixed parameter support.
- Do not break link-expression resolution (`${alias.key}` arithmetic).
- Keep `validate` behavior compatible with existing demos.
- Maintain multiprocessing `if __name__ == "__main__"` guards in scripts that launch optimizers.
- Avoid introducing expensive default settings in demo scripts.

## Quick Sanity Workflow For Agent Changes
1. Run a non-optimization geometry build with simulation disabled first:
   - `build_mifa(params, run_simulation=False, view_skeleton=False)`
2. Validate params explicitly if changing geometry rules:
   - `validate_ifa_params(params)`
3. Only then run short optimization smoke checks (small `maxiter/popsize`).

## Known Gaps / Caveats
- Repository does not currently include an automated test suite.
- README has several stale filenames/typos; prefer script names present in repo.
- Solver installation and runtime environment are external prerequisites.
