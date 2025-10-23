import multiprocessing as mp
from optimize_lib import optimize_ifa, mm  # and anything else you use

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
    'board_hsub': 90*mm,
    'board_th': 1.5*mm,
    'mifa_meander': 2*mm,
    'mifa_meander_edge_distance': 3*mm,
    'mifa_tipdistance': 3*mm,
    'f1': 0.7e9,
    'f0': 0.8e9,
    'f2': 0.9e9,
    'freq_points': 5,
    'mesh_boundry_size_divisor': 0.4,
    'mesh_wavelength_fraction': 0.3,
    'lambda_scale': 0.7,
}

parameters = {
    'board_wsub': 0.021, 
    'board_th': 0.0015,
    'f0': 2.45e+09, 
    'f1': 2.3e+09, 
    'f2': 2.6e+09, 
    'freq_points': 3, 
    'board_hsub': 0.09, 
    'ifa_e': 0.0005, 
    'ifa_e2': 0.000575394784, 
    'ifa_fp': 0.00378423695, 
    'ifa_h': 0.00790411482, 
    'ifa_l': 0.0196761041, 
    'ifa_te': 0.0005, 
    'ifa_w1': 0.000550173526, 
    'ifa_w2': 0.00129312109, 
    'ifa_wf': 0.000433478781, 
    'mesh_boundry_size_divisor': 0.33,
    'mesh_wavelength_fraction': 0.2, 
    'mifa_meander': 0.002, 
    'mifa_meander_edge_distance': 0.003, 
    'mifa_tipdistance': 0.003, 
    'via_size': 0.0005,  
    'lambda_scale': 1 }

# IMPORTANT: set bounds in METERS. Multiply EACH entry by mm.
optimize_parameters = { 
    'ifa_h':  (10.0*mm, 30.0*mm),
    'ifa_l':  (50*mm, 200*mm),
    'ifa_w1': (0.6*mm, 1.5*mm),
    'ifa_w2': (0.6*mm, 1.5*mm),
    'ifa_wf': (0.6*mm, 1.5*mm),
    'ifa_fp': (3.5*mm, 10*mm),
    'ifa_e2':  (0.5*mm, 10*mm),
}
def main():
    best_params, result, summary = optimize_ifa(
        start_parameters=parameters,
        optimize_parameters=optimize_parameters,
        maxiter=30,
        popsize=14,
        seed=1,
        polish=False,
        solver_name="CUDSS",
        timeout=200.0,  
        bandwidth_target_db=-10.0,
        bandwidth_span=(parameters['f1'], parameters['f2']),
        bandwidth_weight=2.0,
        include_start=True,
        start_jitter=0.05,
        log_every_eval=False,
    )
    print("FINAL BEST PARAMS:", best_params)



if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
