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
the ouptut is logged to a folder best_params_log/SIMULATION_NAME_stageX.log

#############################################################
#|------------- substrate_width -------------------|
# _______________________________________________     _ substrate_thickness
#| A  ifa_e      |----------ifa_l(total length)-| |\  \-gndplane_position 
#| V____          _______________     __________  | |  \_0 point
#|               |    ___  ___   |___|  ______  | | |
#|         ifa_h |   |   ||   |_________|    |  |_|_|_ mifa_low_dist 
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
    'ifa_h': 20.0*mm,
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
    'mifa_low_dist': 3*mm,
    'mifa_tipdistance': 3*mm,
    'f1': 0.7e9,
    'f0': 0.8e9,
    'f2': 0.9e9,
    'freq_points': 3,
    'mesh_boundary_size_divisor': 0.4,
    'mesh_wavelength_fraction': 0.4,
    'lambda_scale': 0.5,
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

    p['freq_points'] = 3
    p['lambda_scale'] = 1.0
    p['mesh_wavelength_fraction'] = 0.20
    p['mesh_boundary_size_divisor'] = 0.33

    best_params, result, summary = run_stage(
        f"{SIMULATION_NAME}_broad",
        p, bounds,
        maxiter=3, popsize=100, seed=1,
        solver_name=SOLVER, timeout=200.0,
        bandwidth_target_db=-11.0, bandwidth_span=(p['f1'], p['f2']),
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