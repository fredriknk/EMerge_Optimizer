import multiprocessing as mp
from optimize_lib import optimize_ifa, mm  # and anything else you use
import emerge as em

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
        solver=em.EMSolver.CUDSS,
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
    mp.freeze_support()  # required on Windows for spawn
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set
    main()
