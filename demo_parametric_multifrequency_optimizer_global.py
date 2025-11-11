from typing import Dict, Tuple
import numpy as np
from ifalib2 import mm, build_mifa
from optimize_lib import local_minimize_ifa ,_fmt_params_singleline_raw, write_json,global_optimizer
from datetime import datetime
import os
import multiprocessing as mp
from datetime import datetime

params= { 
        'p.ifa_h': 0.025, 'p.ifa_l': 0.120,
        'p.ifa_w1': 0.001, 'p.ifa_w2': 0.0005, 'p.ifa_wf': 0.0005, 
        'p.board_wsub': 0.03, 
        'p.board_hsub': 0.04,
        'p.board_th': 0.0015, 
        #'p.sweep_freqs': np.array([0.806e9,0.847e+09, 1.7475e9,1.8425e9]), 
        #'p.sweep_weights': np.array([1., 1., 1., 1.]),
        'p.sweep_freqs': np.array([0.826e9,1.81e9]),
        'p.sweep_weights': np.array([1., 1.]),
        'p.ifa_e': 0.0005, 
        'p.ifa_e2': 0.0005, 'p.ifa_fp': 0.002, 
        'p.ifa_te': 0.0005, 
        
        'p.mesh_boundary_size_divisor': 0.33, 
        'p.mesh_wavelength_fraction': 0.2, 'p.mifa_meander': 0.0023, 
        'p.mifa_low_dist': 0.011, 
        'p.mifa_tipdistance': '${p.mifa_low_dist}', 
        'p.via_size': 0.0005, 
        'p.lambda_scale': 1, 
        'p.validate': True,
                
        'p2.ifa_l': 0.060, 'p2.ifa_h': 0.01, 
        'p2.ifa_e': '${p.ifa_fp}', # When using no shunt set ifa_e to ifa_fp to align
        'p2.ifa_w2': 0.001, 
        'p2.mifa_meander': 0.0023, 
        'p2.mifa_low_dist': 0.001, 
        'p2.mifa_tipdistance':  0.001, 
        'p2.shunt': 0,
        'p2.validate': True,}

BASE_BOUNDS: Dict[str, Tuple[float, float]] = {
    'p.ifa_h':  (15*mm, 25.0*mm),
    'p.ifa_l':  (60*mm,   120*mm),
    #'p.ifa_w1': (0.3*mm,  2*mm),
    #'p.ifa_w2': (0.3*mm,  1*mm),
    #'p.ifa_wf': (0.3*mm,  1*mm),
    'p.ifa_fp': (0.6*mm,  10*mm),
    'p.mifa_meander': (1.5*mm, 2.3*mm),
    'p.mifa_low_dist': (5.5*mm, 15*mm),
    
    'p2.ifa_h':  (3*mm, 15.0*mm),
    'p2.ifa_l':  (20*mm,   80*mm),
    #'p2.ifa_w2': (0.3*mm,  1*mm),
    'p2.mifa_meander': (1*mm, 2.3*mm),
    'p2.mifa_low_dist': (0.3*mm, 5*mm),
}

SOLVER = "PARDISO"
SOLVER = "CUDSS"

SIMULATION_NAME = f"{datetime.now().strftime('%Y%m%d_%H%M')}_Mifa_multifreq_optimization800mhz_1800mhz_global"

def main():
    p = dict(params)
    bounds = dict(BASE_BOUNDS)
    
    #build_mifa(params,view_skeleton=True,run_simulation=False)  # Uncomment to test build once before optimization
    
    p['p.lambda_scale'] = 0.6
    p['p.mesh_wavelength_fraction'] = 0.20
    p['p.mesh_boundary_size_divisor'] = 0.33

    
    best_local, result, summary = global_optimizer(
        f"{datetime.now().strftime('%Y%m%d_%H%M')}_{SIMULATION_NAME}_global",
        p, bounds,
        maxiter=1, popsize=3000, seed=3,
        bandwidth_target_db=None, bandwidth_span=None,
        solver_name=SOLVER, timeout=300.0,
        include_start=False, log_every_eval=True
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