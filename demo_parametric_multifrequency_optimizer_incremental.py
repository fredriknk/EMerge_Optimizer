import numpy as np
from ifalib2 import mm, normalize_params_sequence,denormalize_params_sequence_flat
from optimize_lib import local_minimize_ifa ,_fmt_params_singleline_raw, write_json
from datetime import datetime
import os
import multiprocessing as mp
from numpy import array

params = { 
          'p.board_wsub': 0.0191, 'p.board_th': 0.0015,  'p.board_hsub': 0.06,
          'p.sweep_freqs': np.array([2.45e+09, 5.80e+09]), 'p.sweep_weights': np.array([1., 1.]), 
          'p.ifa_e': 0.0005, 'p.ifa_e2': 0.000575394784, 'p.ifa_te': 0.0005,
          'p.ifa_fp': 0.00272626454, 'p.ifa_h': 0.00829488465, 'p.ifa_l': 0.0234122232, 
          'p.ifa_w1': 0.000435609747, 'p.ifa_w2': 0.000537390996, 'p.ifa_wf': 0.00062670221, 
          'p.mesh_boundary_size_divisor': 0.33, 'p.mesh_wavelength_fraction': 0.2, 
          'p.mifa_meander': 0.0023, 'p.mifa_low_dist': '${p.ifa_h} - 0.003', 
          'p.mifa_tipdistance': '${p.mifa_low_dist}', 'p.via_size': 0.0005, 
          'p.lambda_scale': 1, 
          
          'p2.ifa_l': 0.00601520693, 
          'p2.ifa_h': '${p.mifa_low_dist}- 0.0005', 'p2.ifa_e': '${p.ifa_fp}', 
          'p2.ifa_w2': 0.000388412941, 'p2.mifa_meander': '${p2.ifa_w2}*2+0.0003', 
          'p2.mifa_low_dist': '${p.mifa_low_dist} - 0.003', 
          'p2.mifa_tipdistance': '${p2.mifa_low_dist}', 
          'p2.shunt': 0 }

params = { 'p.ifa_h': 0.0237162003, 'p.ifa_l': 0.106715948, 'p.ifa_w1': 0.001, 'p.ifa_w2': 0.0005, 'p.ifa_wf': 0.0005, 'p.board_wsub': 0.03, 'p.board_hsub': 0.04, 'p.board_th': 0.0015, 'p.sweep_freqs': array([8.0600e+08, 8.4700e+08, 1.7475e+09, 1.8425e+09]), 'p.sweep_weights': array([1., 1., 1., 1.]), 'p.ifa_e': 0.0005, 'p.ifa_e2': 0.0005, 'p.ifa_fp': 0.00195366811, 'p.ifa_te': 0.0005, 'p.mesh_boundary_size_divisor': 0.33, 'p.mesh_wavelength_fraction': 0.2, 'p.mifa_meander': 0.00174759215, 'p.mifa_low_dist': 0.0092598482, 'p.mifa_tipdistance': '${p.mifa_low_dist}', 'p.via_size': 0.0005, 'p.lambda_scale': 0.6, 'p.validate': 1, 'p2.ifa_l': 0.0205744124, 'p2.ifa_h': 0.00574348287, 'p2.ifa_e': '${p.ifa_fp}', 'p2.ifa_w2': 0.001, 'p2.mifa_meander': 0.00218982907, 'p2.mifa_low_dist': 0.00327758215, 'p2.mifa_tipdistance': 0.001, 'p2.shunt': 0, 'p2.validate': 1 }


normalize_parameters = normalize_params_sequence(params)
params = denormalize_params_sequence_flat(normalize_parameters)


tweak_parameters = [
    'p.ifa_l',
    'p.ifa_fp',
    'p.ifa_h',
    'p.ifa_w1',
    'p.ifa_w2',
    'p.ifa_e2',
    'p.mifa_meander',
    'p.ifa_wf',
    'p2.ifa_l',
    'p2.ifa_w2',
    'p2.ifa_h',
    'p2.mifa_meander',
    'p2.mifa_low_dist',
    'p2.mifa_tipdistance',
]

base_bounds = {}
span = 0.20  # +/- 20%
for k in list(params.keys()):
    if k in tweak_parameters:
        val = params[k]
        delta = val * span
        base_bounds[k] = (val - delta, val + delta)

SOLVER = "PARDISO"
SOLVER = "CUDSS"

SIMULATION_NAME = "mifa_800mhz_1800mhz_optimization_incremental"

def main():
    p = dict(params)
    bounds = dict(base_bounds)
    
    
    p['p.lambda_scale'] = 0.6
    p['p.mesh_wavelength_fraction'] = 0.3
    p['p.mesh_boundary_size_divisor'] = 0.5
    

    best_local, sum_local = local_minimize_ifa(
        start_parameters=p,            # your seed (e.g., current best)
        optimize_parameters=bounds,      # bounds (meters)
        method="Powell",                         # or "Nelder-Mead"
        init_step_mm=0.1,                       # “small step” knob
        maxiter=1000,
        solver_name="CUDSS",
        stage_name=f"{datetime.now().strftime('%Y%m%d_%H%M')}_{SIMULATION_NAME}",
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