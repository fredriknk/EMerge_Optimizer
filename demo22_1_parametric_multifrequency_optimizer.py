import numpy as np
from ifalib import mm
from optimize_lib import optimize_multifreq_nested
from datetime import datetime

from demo22_0_parametric_multifrequency_mifa2 import params  # or just copy the dict

params ={
    "p": {
        'board_wsub': 0.0191, 
        'board_th': 0.0015,
        'sweep_freqs': np.array([2.45e9,5.0e9]),
        'sweep_weights': np.array([1.0,1.0]),
        'board_hsub': 0.06, 
        'ifa_e': 0.0005, 
        'ifa_e2': 0.000575394784, 
        'ifa_fp': 0.0045, 
        'ifa_h': 0.00790411482, 
        'ifa_l': 0.020, 
        'ifa_te': 0.0005, 
        'ifa_w1': 0.0005, 
        'ifa_w2': 0.001, 
        'ifa_wf': 0.0005, 
        'mesh_boundary_size_divisor': 0.33,
        'mesh_wavelength_fraction': 0.2, 
        'mifa_meander': 0.001*2+0.0003, 
        'mifa_low_dist': "${p.ifa_h} - 0.003", 
        'mifa_tipdistance': "${p.mifa_low_dist}", 
        'via_size': 0.0005,  
        'lambda_scale': 1 
    },
    "p2" : {
        "ifa_l":0.021,
        "ifa_h":"${p.mifa_low_dist}- 0.0005",
        "ifa_e":"${p.ifa_fp}",
        "ifa_w2":0.0006,
        "mifa_meander":"${p2.ifa_w2}*2+0.0003",
        "mifa_low_dist":"${p.mifa_low_dist} - 0.003",
        "mifa_tipdistance":"${p2.mifa_low_dist}",
        "shunt":False,
    },
}

BASE_BOUNDS = {
    "p": {
        'ifa_h':        (6*mm, 12.0*mm),
        'ifa_l':        (12*mm, 36*mm),
        'ifa_w1':       (0.3*mm, 2*mm),
        'ifa_w2':       (0.3*mm, 1*mm),
        'ifa_wf':       (0.3*mm, 1*mm),
        'ifa_fp':       (3*mm, 12*mm),
    },
    "p2": {
        'ifa_l':        (5*mm, 36*mm),
        'ifa_w2':       (0.3*mm, 1*mm),
    },
}
STAGE_NAME = "mifa_multifreq_demo22"
if __name__ == "__main__":
    best_params, result, summary = optimize_multifreq_nested(
        start_parameters=params,
        bounds_nested=BASE_BOUNDS,
        maxiter=2,
        popsize=80,
        seed=123,
        solver_name="CUDSS",
        timeout=900.0,
        include_start=True,
        log_every_eval=False,
        stage_name=f"{datetime.now().strftime('%Y%m%d')}_{STAGE_NAME}",
    )

    print("SUCCESS:", summary["optimizer_success"])
    print("Best sweep RL:")
    for f, rl in zip(summary["best_sweep_freqs_Hz"], summary["best_RL_dB_at_sweep"]):
        print(f"  {rl:6.2f} dB @ {f/1e9:.4f} GHz")
