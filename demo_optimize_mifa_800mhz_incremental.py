import json, os, math
import multiprocessing as mp
from typing import Dict, Tuple
from optimize_lib import global_optimizer,shrink_bounds_around_best, write_json, mm, _fmt_params_singleline_raw, OptLogger,local_minimize_ifa
""" MIFA OPTIMIZATION DEMO

In this demo we build mifa antenna geometry and optimize it for operation
around 800MHz with goals for low reflection and wide bandwidth.

This simulation is very heavy and might take a while to fully compute.
Its very reccomended to use a CUDA capable solver for this demo.

The optimizer spawns single simulations to isolate from native chrashes
the ouptut is logged to a folder best_params_log/SIMULATION_NAME_stageX.log

"""

parameters = { 
    'ifa_h': 0.0230161676, 'ifa_l': 0.09415, 
    'ifa_w1': 0.00045360957, 'ifa_w2': 0.000439352308, 'ifa_wf': 0.000411214503, 
    'ifa_fp': 0.00722339282, 'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_te': 0.0005, 
    'via_size': 0.0005, 'board_wsub': 0.03, 'board_hsub': 0.11, 'board_th': 0.0015, 
    'mifa_meander': 0.00169312729, 'mifa_low_dist': 0.003, 'mifa_tipdistance': 0.003,
    'f1': 791000000, 'f0': 826000000, 'f2': 862000000, 'freq_points': 3, 'eps_r': 4.5,
    'mesh_boundary_size_divisor': 0.33, 'mesh_wavelength_fraction': 0.2, 'lambda_scale': 1, 
    }
tweak_parameters = ['ifa_l', 'ifa_h', 'ifa_w1', 'ifa_w2', 'ifa_wf', 'ifa_fp', 'mifa_low_dist', 'mifa_meander']

base_bounds ={}
span = 0.2
#add parameter intervals +/- span % around initial parameters
for k in list(parameters.keys()):
    if k in tweak_parameters:
        val = parameters[k]
        delta = val * span
        base_bounds[k] = (val - delta, val + delta)

SOLVER = "PARDISO"
SOLVER = "CUDSS"

SIMULATION_NAME = "mifa_800mhz_incremental_optimization"
OPTIMIZATION_MODE = "S11_OPTIMIZE"  # "BW_OPTIMIZE" or "S11_OPTIMIZE"

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


    best_local, sum_local = local_minimize_ifa(
        start_parameters=p,              # your seed (e.g., current best)
        optimize_parameters=bounds,      # bounds (meters)
        method="Powell",                         # or "Nelder-Mead"
        init_step_mm=0.05,                       # “small step” knob
        maxiter=1000,
        bandwidth_target_db=bw_target_db, 
        bandwidth_span=bw_span,           
        solver_name="CUDSS",
        stage_name=SIMULATION_NAME,
        log_every_eval=True,
        bandwidth_parameters=bandwidth_parameters,
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