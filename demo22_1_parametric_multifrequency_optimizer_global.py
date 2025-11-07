from typing import Dict, Tuple
import numpy as np
from ifalib import mm
from optimize_lib import local_minimize_ifa ,_fmt_params_singleline_raw, write_json,run_stage
from datetime import datetime
import os
import multiprocessing as mp

params= { 'p.board_wsub': 0.0191, 
         'p.board_th': 0.0015, 
         'p.sweep_freqs': np.array([2.45e+09, 5.80e+09]), 
         'p.sweep_weights': np.array([1., 1.]), 
         'p.board_hsub': 0.06, 'p.ifa_e': 0.0005, 
         'p.ifa_e2': 0.000575394784, 'p.ifa_fp': 0.00364461081, 
         'p.ifa_h': 0.00909984885, 'p.ifa_l': 0.0307, 
         'p.ifa_te': 0.0005, 
         'p.ifa_w1': 0.00112657281, 'p.ifa_w2': 0.000445781771, 'p.ifa_wf': 0.000398836163, 
         'p.mesh_boundary_size_divisor': 0.33, 
         'p.mesh_wavelength_fraction': 0.2, 'p.mifa_meander': 0.0023, 
         'p.mifa_low_dist': '${p.ifa_h} - 0.003', 
         'p.mifa_tipdistance': '${p.mifa_low_dist}', 
         'p.via_size': 0.0005, 
         'p.lambda_scale': 1, 
         
         'p2.ifa_l': 0.00796519359, 'p2.ifa_h': '${p.mifa_low_dist}- 0.0005', 
         'p2.ifa_e': '${p.ifa_fp}', # When using no shunt set ifa_e to ifa_fp to align
         'p2.ifa_w2': 0.00033083229, 
         'p2.mifa_meander': '${p2.ifa_w2}*2+0.0003', 
         'p2.mifa_low_dist': '${p.mifa_low_dist} - 0.003', 
         'p2.mifa_tipdistance': '${p2.mifa_low_dist}', 
         'p2.shunt': 0 }

BASE_BOUNDS: Dict[str, Tuple[float, float]] = {
    'p.ifa_h':  (6*mm, 12.0*mm),
    'p.ifa_l':  (5*mm,   36*mm),
    'p.ifa_w1': (0.3*mm,  2*mm),
    'p.ifa_w2': (0.3*mm,  1*mm),
    'p.ifa_wf': (0.3*mm,  1*mm),
    'p.ifa_fp': (0.6*mm,  10*mm),
    
    'p2.ifa_l':  (5*mm,   36*mm),
    'p2.ifa_w2': (0.3*mm,  1*mm),
}

SOLVER = "PARDISO"
SOLVER = "CUDSS"

SIMULATION_NAME = "Mifa_multifreq_optimization2_45_5_8_global"

def main():
    p = dict(params)
    bounds = dict(BASE_BOUNDS)
    
    
    p['p.lambda_scale'] = 1.0
    p['p.mesh_wavelength_fraction'] = 0.20
    p['p.mesh_boundary_size_divisor'] = 0.33

    
    best_local, result, summary = run_stage(
        f"{datetime.now().strftime('%Y%m%d_%H%M')}_{SIMULATION_NAME}_global",
        p, bounds,
        maxiter=1, popsize=300, seed=1,
        bandwidth_target_db=None, bandwidth_span=None,
        solver_name=SOLVER, timeout=200.0,
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