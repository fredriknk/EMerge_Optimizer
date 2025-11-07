import json, os, math
import multiprocessing as mp
from typing import Dict, Tuple
from optimize_lib import global_optimizer,shrink_bounds_around_best, write_json, mm, _fmt_params_singleline_raw, OptLogger,local_pattern_search_ifa , local_minimize_ifa
""" MIFA OPTIMIZATION DEMO

The optimizer takes a good starting point and does local optimization with
rewards for minimal reflection across the band, average min reflection and center 
frequency.

This simulation is very heavy and might take a while to fully compute.
Its very reccomended to use a CUDA capable solver for this demo.

The optimizer spawns single simulations to isolate from native chrashes
the ouptut is logged to a folder best_params_log/SIMULATION_NAME_stageX.log
"""
# Initial parameters - start point for optimization

parameters = mifa_14x25_2450mhz = { 
    'ifa_h': 0.00773189309, 'ifa_l': 0.0229509148, 
    'ifa_w1': 0.000766584703, 'ifa_w2': 0.000440876843, 'ifa_wf': 0.000344665757, 
    'ifa_fp': 0.00156817497, 'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_te': 0.0005, 
    'via_size': 0.0003, 'board_wsub': 0.014, 'board_hsub': 0.025, 'board_th': 0.0015, 
    'mifa_meander': 0.00195527223, 'mifa_low_dist': 0.00217823618, 
    'f1': 2.4e+09, 'f0': 2.45e+09, 'f2': 2.5e+09, 'freq_points': 3, 
    'mesh_boundary_size_divisor': 0.33, 'mesh_wavelength_fraction': 0.2, 'lambda_scale': 1 }

epsilon_r = 4.4  # FR4 typical
calc_wavelength_at_2_45ghz = (3e8 / 2.45e9)*epsilon_r**0.5
# IMPORTANT: set bounds in METERS

tweak_parameters = ['ifa_h', 'ifa_l', 'ifa_w1', 'ifa_w2', 'ifa_wf', 'ifa_fp', 'mifa_low_dist', 'mifa_meander']

base_bounds ={}
#add parameter intervals +/- 10% around initial parameters
for k in list(parameters.keys()):
    if k in tweak_parameters:
        val = parameters[k]
        delta = val * 0.10
        base_bounds[k] = (val - delta, val + delta)

SOLVER = "PARDISO"
SOLVER = "CUDSS"

SIMULATION_NAME = "mifa_2400mhz_optimization_incremental"
OPTIMIZATION_MODE = "BW_OPTIMIZE"

def main():
    p = dict(parameters)
    bounds = dict(base_bounds)
    
    if OPTIMIZATION_MODE == "BW_OPTIMIZE":
        bw_span = (p['f1'], p['f2'])
        bw_target_db = 10.0
        bandwidth_parameters = {
            "mean_excess_weight": 1.0,      #Average excess reflection weight
            "max_excess_factor": 0.1,       #lowest-case excess reflection weight
            "center_weighting_factor": 0.05, #Center frequency reflection weight
            "mean_power_weight": 0.025,       #Mean power reflection weight    
        }
        
        p['freq_points'] = 3
    else: # S11_OPTIMIZE
        bw_span = None
        bw_target_db = None
        bandwidth_parameters = None
        p['f1'] = p['f0'] 
        p['f2'] = p['f0']
        p['freq_points'] = 1
    
    
    p['lambda_scale'] = 1.0
    p['mesh_wavelength_fraction'] = 0.20
    p['mesh_boundary_size_divisor'] = 0.33
    

    best_local, sum_local = local_minimize_ifa(
        start_parameters=p,            # your seed (e.g., current best)
        optimize_parameters=bounds,      # bounds (meters)
        method="Powell",                         # or "Nelder-Mead"
        init_step_mm=0.05,                       # “small step” knob
        maxiter=1000,
        bandwidth_target_db=10.0,               # Comment out to disable bandwidth goal
        bandwidth_parameters=bandwidth_parameters,
        bandwidth_span=(p['f1'], p['f2']),      # Comment out to disable bandwidth goal
        solver_name="CUDSS",
        stage_name=SIMULATION_NAME,
        log_every_eval=True,
    )

    # Done: save final winner, print compact line again
    os.makedirs("best_params_log", exist_ok=True)
    write_json(f"best_params_log/best_params_{SIMULATION_NAME}.json", best_local)
    print("\n=== FINAL WINNER ===")
    print(_fmt_params_singleline_raw(best_local, sort_keys=False))

if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()