import json, os, math
import multiprocessing as mp
from typing import Dict, Tuple
from optimize_lib import run_stage,shrink_bounds_around_best, write_json, mm, _fmt_params_singleline_raw, OptLogger
""" MIFA OPTIMIZATION DEMO

In this demo we build mifa antenna geometry and optimize it for operation
around 800MHz with goals for low reflection and wide bandwidth.

This simulation is very heavy and might take a while to fully compute.
Its very reccomended to use a CUDA capable solver for this demo.

The optimizer spawns single simulations to isolate from native chrashes
the ouptut is logged to 

#############################################################
#|------------- substrate_width -------------------|
# _______________________________________________     _ substrate_thickness
#| A  ifa_e      |----------ifa_l(total length)-| |\  \-gndplane_position 
#| V____          _______________     __________  | |  \_0 point
#|               |    ___  ___   |___|  ______  | | |
#|         ifa_h |   |   ||   |_________|    |  |_|_|_ mifa_meander_edge_distance 
#|               |   |   ||  mifa_meander    |__|_|_|_ mifa_tipdistance
#|               |   |   ||                   w2  | | |                  
#|_______________|___|___||_______________________| |_|
#| <---ifa_e---->| w1|   wf\                      | |
#|               |__fp___|  \                     | |
#|                       |    feed point          | |
#|                       |                        | | substrate_length
#|<- substrate_width/2 ->|                        | |
#|                                                | |
#|________________________________________________| |
# \________________________________________________\|
#############################################################
Note: ifa_l is total length including meanders and tip
"""

parameters = { 
    'ifa_h': 10.0*mm,
    'ifa_l': 100*mm,
    'ifa_w1': 0.619*mm,
    'ifa_w2': 0.5*mm,
    'ifa_wf': 0.5*mm,
    'ifa_fp': 2*mm,
    'ifa_e': 0.5*mm,
    'ifa_e2': 0.5*mm,
    'ifa_te': 0.5*mm,
    'via_size': 0.5*mm,
    'board_wsub': 30*mm,
    'board_hsub': 110*mm,
    'board_th': 1.5*mm,
    'mifa_meander': 2*mm,
    'mifa_meander_edge_distance': 3*mm,
    'mifa_tipdistance': 3*mm,
    'f1': 0.7e9,
    'f0': 0.8e9,
    'f2': 0.9e9,
    'freq_points': 3,
    'mesh_boundry_size_divisor': 0.4,
    'mesh_wavelength_fraction': 0.4,
    'lambda_scale': 0.5,
}

parameters = {
  "ifa_h": 0.026721961022660216,
  "ifa_l": 0.13516767907510804,
  "ifa_w1": 0.0007747970923122423,
  "ifa_w2": 0.0008121799266081413,
  "ifa_wf": 0.0012112860858232545,
  "ifa_fp": 0.007857536003039053,
  "ifa_e": 0.0005,
  "ifa_e2": 0.0005,
  "ifa_te": 0.0005,
  "via_size": 0.0005,
  "board_wsub": 0.03,
  "board_hsub": 0.11,
  "board_th": 0.0015,
  "mifa_meander": 0.002,
  "mifa_meander_edge_distance": 0.003,
  "f1": 791000000.0,
  "f0": 826000000.0,
  "f2": 862000000.0,
  "freq_points": 3.0,
  "mesh_boundry_size_divisor": 0.5,
  "mesh_wavelength_fraction": 0.5,
  "lambda_scale": 0.5,
  "clearance": 0.0003,
}




# IMPORTANT: set bounds in METERS
BASE_BOUNDS: Dict[str, Tuple[float, float]] = {
    'ifa_h':  (10.0*mm, 35.0*mm),
    'ifa_l':  (105*mm,   135*mm),
    'ifa_w1': (0.6*mm,  2*mm),
    'ifa_w2': (0.6*mm,  1*mm),
    'ifa_wf': (0.6*mm,  1*mm),
    'ifa_fp': (2*mm,  12*mm),
    'ifa_mifa_meander_edge_distance': (2*mm, 20*mm),
    "mifa_meander": (1.2*mm, 3*mm),
}


SOLVER = "PARDISO"
SOLVER = "CUDSS"

SIMULATION_NAME = "mifa_800mhz_optimization"

def main():
    # Keep an independent copy we can mutate stage-by-stage
    p = dict(parameters)
    bounds = dict(BASE_BOUNDS)

    # ----------------- Stage 0: Quicksearch (very fast & coarse) -----------------
    # Coarse mesh / fewer points / modest λ_scaling
    p['freq_points'] = 3
    p['lambda_scale'] = 0.5
    p['mesh_wavelength_fraction'] = 0.50
    p['mesh_boundry_size_divisor'] = 0.50

    run_stage(
        f"{SIMULATION_NAME}_quick",
        p, bounds,
        maxiter=4, popsize=20, seed=99,
        solver_name=SOLVER, timeout=120.0,
        bandwidth_target_db=-10.0, bandwidth_span=(p['f1'], p['f2']), bandwidth_weight=30.0,
        include_start=False, start_jitter=0.05, log_every_eval=False
    )

    # ----------------- Stage 1: Refine (narrow bounds, better mesh) --------------
    bounds = shrink_bounds_around_best(p, bounds, shrink=0.35, min_span_mm=0.05)
    p['freq_points'] = 3
    p['lambda_scale'] = 0.7
    p['mesh_wavelength_fraction'] = 0.30
    p['mesh_boundry_size_divisor'] = 0.40

    run_stage(
        f"{SIMULATION_NAME}_refine1",
        p, bounds,
        maxiter=8, popsize=6, seed=2,
        solver_name=SOLVER, timeout=150.0,
        bandwidth_target_db=-10.0, bandwidth_span=(p['f1'], p['f2']), bandwidth_weight=30.0,
        include_start=True, start_jitter=0.03, log_every_eval=False
    )

    # ----------------- Stage 2: Refine deeper (tight bounds, denser sweep) -------
    bounds = shrink_bounds_around_best(p, bounds, shrink=0.30, min_span_mm=0.03)
    p['freq_points'] = 5
    p['lambda_scale'] = 1.0
    p['mesh_wavelength_fraction'] = 0.20
    p['mesh_boundry_size_divisor'] = 0.33

    best_params, result, summary = run_stage(
        f"{SIMULATION_NAME}_refine2",
        p, bounds,
        maxiter=10, popsize=4, seed=3,
        solver_name=SOLVER, timeout=180.0,
        bandwidth_target_db=-10.0, bandwidth_span=(p['f1'], p['f2']), bandwidth_weight=30.0,
        include_start=True, start_jitter=0.02, log_every_eval=False
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