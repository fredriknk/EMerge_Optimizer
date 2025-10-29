import json, os, math
import multiprocessing as mp
from typing import Dict, Tuple
from optimize_lib import run_stage,shrink_bounds_around_best, write_json, mm, _fmt_params_singleline_raw, OptLogger
""" MIFA OPTIMIZATION DEMO

In this demo we build mifa antenna geometry and optimize it for operation
around 2450MHz with goals for low reflection and wide bandwidth.

This simulation is very heavy and might take a while to fully compute.
Its very reccomended to use a CUDA capable solver for this demo.

The optimizer spawns single simulations to isolate from native chrashes
the ouptut is logged to a folder best_params_log/SIMULATION_NAME_stageX.log

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
    'mifa_meander_edge_distance': 0.5*mm,
    'f1': 2.4e9,
    'f0': 2.45e9,
    'f2': 2.5e9,
    'freq_points': 3,
    'mesh_boundry_size_divisor': 0.4,
    'mesh_wavelength_fraction': 0.4,
    'lambda_scale': 0.5,
}

epsilon_r = 4.4  # FR4 typical
calc_wavelength_at_2_45ghz = (3e8 / 2.45e9)*epsilon_r**0.5
# IMPORTANT: set bounds in METERS
BASE_BOUNDS: Dict[str, Tuple[float, float]] = {
    'ifa_h':  (3.0*mm, 8.0*mm),
    'ifa_l':  (17*mm,   36*mm),
    'ifa_w1': (0.3*mm,  2*mm),
    'ifa_w2': (0.3*mm,  1*mm),
    'ifa_wf': (0.3*mm,  1*mm),
    'ifa_fp': (0.6*mm,  6*mm),
    'mifa_meander_edge_distance': (0.5*mm, 5*mm),
    "mifa_meander": (0.6*mm, 2*mm),
}


SOLVER = "PARDISO"
SOLVER = "CUDSS"

SIMULATION_NAME = "mifa_2400mhz_optimization_single"

def main():
    # Keep an independent copy we can mutate stage-by-stage
    p = dict(parameters)
    bounds = dict(BASE_BOUNDS)

    p['freq_points'] = 3
    p['lambda_scale'] = 1.0
    p['mesh_wavelength_fraction'] = 0.20
    p['mesh_boundry_size_divisor'] = 0.33

    best_params, result, summary = run_stage(
        f"{SIMULATION_NAME}_refine2",
        p, bounds,
        maxiter=10, popsize=30, seed=4,
        solver_name=SOLVER, timeout=200.0,
        bandwidth_target_db=-10.0, bandwidth_span=(p['f1'], p['f2']),
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