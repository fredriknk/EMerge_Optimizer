import json, os, math
import multiprocessing as mp
from typing import Dict, Tuple
from optimize_lib import run_stage,shrink_bounds_around_best, write_json, mm, _fmt_params_singleline_raw, OptLogger,local_pattern_search_ifa , local_minimize_ifa
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
# Initial parameters - start point for optimization
parameters={ 'ifa_h': 0.00789656661, 'ifa_l': 0.0229509148, 'ifa_w1': 0.000766584703, 'ifa_w2': 0.000440876843, 'ifa_wf': 0.000344665757, 'ifa_fp': 0.00156817497, 'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_te': 0.0005, 'via_size': 0.0003, 'board_wsub': 0.014, 'board_hsub': 0.025, 'board_th': 0.0015, 'mifa_meander': 0.00195527223, 'mifa_meander_edge_distance': 0.00217823618, 'f1': 2.4e+09, 'f0': 2.45e+09, 'f2': 2.5e+09, 'freq_points': 3, 'mesh_boundry_size_divisor': 0.33, 'mesh_wavelength_fraction': 0.2, 'lambda_scale': 1 }
parameters = mifa_14x25_2450mhz = { 
    'ifa_h': 0.00773189309, 'ifa_l': 0.0229509148, 
    'ifa_w1': 0.000766584703, 'ifa_w2': 0.000440876843, 'ifa_wf': 0.000344665757, 
    'ifa_fp': 0.00156817497, 'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_te': 0.0005, 
    'via_size': 0.0003, 'board_wsub': 0.014, 'board_hsub': 0.025, 'board_th': 0.0015, 
    'mifa_meander': 0.00195527223, 'mifa_meander_edge_distance': 0.00217823618, 
    'f1': 2.4e+09, 'f0': 2.45e+09, 'f2': 2.5e+09, 'freq_points': 3, 
    'mesh_boundry_size_divisor': 0.33, 'mesh_wavelength_fraction': 0.2, 'lambda_scale': 1 }


epsilon_r = 4.4  # FR4 typical
calc_wavelength_at_2_45ghz = (3e8 / 2.45e9)*epsilon_r**0.5
# IMPORTANT: set bounds in METERS

tweak_parameters = ['ifa_h', 'ifa_l', 'ifa_w1', 'ifa_w2', 'ifa_wf', 'ifa_fp', 'mifa_meander_edge_distance', 'mifa_meander']

base_bounds ={}
#add parameter intervals +/- 10% around initial parameters
for k in list(parameters.keys()):
    if k in tweak_parameters:
        val = parameters[k]
        delta = val * 0.10
        base_bounds[k] = (val - delta, val + delta)

SOLVER = "PARDISO"
SOLVER = "CUDSS"

SIMULATION_NAME = "mifa_2400mhz_optimization_single"

def main():
    # Keep an independent copy we can mutate stage-by-stage
    p = dict(parameters)
    bounds = base_bounds
    p['f1'] = p['f0']
    p['f2'] = p['f0']
    p['freq_points'] = 1
    p['lambda_scale'] = 1.0
    p['mesh_wavelength_fraction'] = 0.20
    p['mesh_boundry_size_divisor'] = 0.33
    

    best_local, sum_local = local_minimize_ifa(
        start_parameters=p,            # your seed (e.g., current best)
        optimize_parameters=bounds,      # bounds (meters)
        method="Powell",                         # or "Nelder-Mead"
        init_step_mm=0.05,                       # “small step” knob
        maxiter=120,
        #bandwidth_target_db=10.0,               # Comment out to disable bandwidth goal
        #bandwidth_span=(p['f1'], p['f2']),      # Comment out to disable bandwidth goal
        solver_name="CUDSS",
        stage_name="powell_local"
    )

    # Done: save final winner, print compact line again
    write_json("best_params.json", best_local)
    print("\n=== FINAL WINNER ===")
    print(_fmt_params_singleline_raw(best_local, sort_keys=False))

if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()