from datetime import datetime
import json, os, math
import multiprocessing as mp
from typing import Dict, Tuple
from optimize_lib import global_optimizer,shrink_bounds_around_best, write_json, mm, _fmt_params_singleline_raw, OptLogger
""" MIFA OPTIMIZATION DEMO

In this we do a wide search for good MIFA antenna parameters to use in
the incremental optimization demo demo20_1_optimize_mifa_2450mhz_incremental.py
The optimizer looks for candidates with f1-f2 refection under -10dB with a small
reward for minimal reflection across the band, average min reflection and center 
frequency.

This simulation is very heavy and might take a while to fully compute.
Its very reccomended to use a CUDA capable solver for this demo.

The optimizer spawns single simulations to isolate from native chrashes
the ouptut is logged to a folder best_params_log/SIMULATION_NAME_stageX.log
"""

parameters = { 
    'ifa_h': 6.0*mm,
    'ifa_l': 20*mm,
    'ifa_w1': 0.5*mm,
    'ifa_w2': 0.5*mm,
    'ifa_wf': 0.5*mm,
    'ifa_fp': 2*mm,
    'ifa_e': 0.5*mm,
    'ifa_e2': 0.5*mm,
    'ifa_te': 0.5*mm,
    'via_size': 0.3*mm,
    'board_wsub': 14*mm,
    'board_hsub': 25*mm,
    'board_th': 1.5*mm,
    'mifa_meander': 1.5*mm,
    'mifa_low_dist': 0.5*mm,
    'f1': 2.4e9,
    'f0': 2.45e9,
    'f2': 2.5e9,
    'freq_points': 3,
    'mesh_boundary_size_divisor': 0.4,
    'mesh_wavelength_fraction': 0.4,
    'lambda_scale': 0.5,
}

epsilon_r = 4.4  # FR4 typical
calc_wavelength_at_2_45ghz = (3e8 / 2.45e9)*epsilon_r**0.5
# IMPORTANT: set bounds in METERS
# Set min and max for each parameter to be optimized
BASE_BOUNDS: Dict[str, Tuple[float, float]] = {
    'ifa_h':  (3.0*mm, 8.0*mm),
    'ifa_l':  (17*mm,   36*mm),
    'ifa_w1': (0.3*mm,  2*mm),
    'ifa_w2': (0.3*mm,  1*mm),
    'ifa_wf': (0.3*mm,  1*mm),
    'ifa_fp': (0.6*mm,  6*mm),
    'mifa_low_dist': (0.5*mm, 5*mm),
    "mifa_meander": (0.6*mm, 2*mm),
}

bandwidth_parameters = {
    "mean_excess_weight": 1.0,      #Average excess reflection weight
    "max_excess_factor": 0.1,       #lowest-case excess reflection weight
    "center_weighting_factor": 0.05, #Center frequency reflection weight
    "mean_power_weight": 0.025,       #Mean power reflection weight    
}


SOLVER = "PARDISO"
SOLVER = "CUDSS"

SIMULATION_NAME = "mifa_2400mhz_optimization_Global"

def main():
    # Keep an independent copy we can mutate stage-by-stage
    p = dict(parameters)
    bounds = dict(BASE_BOUNDS)

    p['freq_points'] = 3
    p['lambda_scale'] = 1.0
    p['mesh_wavelength_fraction'] = 0.20
    p['mesh_boundary_size_divisor'] = 0.33

    best_params, result, summary = global_optimizer(
        f"{datetime.now().strftime('%Y%m%d_%H%M')}_{SIMULATION_NAME}",
        p, bounds,
        maxiter=3, popsize=100, seed=1,
        solver_name=SOLVER, timeout=200.0,
        # bandwidth_target_db=-11.0, bandwidth_span=(p['f1'], p['f2']),
        # bandwidth_parameters=bandwidth_parameters,
        include_start=False, log_every_eval=True
    )

    # Done: save final winner, print compact line again
    write_json("best_params.json", summary["best_params"])
    print("\n=== FINAL WINNER ===")
    print(_fmt_params_singleline_raw(summary["best_params"], sort_keys=False))

if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()