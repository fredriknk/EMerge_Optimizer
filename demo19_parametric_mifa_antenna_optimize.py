# run_optimize.py
import sys, time, traceback, faulthandler
faulthandler.enable()

from optimize_lib import optimize_ifa, _fmt_params_singleline_mm
mm = 1e-3

def main():
    parameters = { 'boundry_size_divisor': 0.4, 'f0': 2.45e+09, 'f1': 2.3e+09, 'f2': 2.6e+09, 'freq_points': 3, 'hsub': 0.09, 'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_fp': 0.00459608786, 'ifa_h': 0.00743038264, 'ifa_l': 0.0211784476, 'ifa_te': 0.0005, 'ifa_w1': 0.000641190557, 'ifa_w2': 0.00138621391, 'ifa_wf': 0.000434786982, 'mifa_meander': 0.002, 'mifa_meander_edge_distance': 0.003, 'mifa_tipdistance': 0.003, 'th': 0.0015, 'via_size': 0.0005, 'wavelength_fraction': 0.3, 'wsub': 0.021, 'lambda_scale': 1 }
    parameters = { 'boundry_size_divisor': 0.4, 'f0': 2.45e+09, 'f1': 2.3e+09, 'f2': 2.6e+09, 'freq_points': 3, 'hsub': 0.09, 'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_fp': 0.00435061745, 'ifa_h': 0.00759614848, 'ifa_l': 0.0207929428, 'ifa_te': 0.0005, 'ifa_w1': 0.000542306047, 'ifa_w2': 0.00139626179, 'ifa_wf': 0.000441327827, 'mifa_meander': 0.002, 'mifa_meander_edge_distance': 0.003, 'mifa_tipdistance': 0.003, 'th': 0.0015, 'via_size': 0.0005, 'wavelength_fraction': 0.3, 'wsub': 0.021, 'lambda_scale': 1 }
 
    optimize_parameters = {
        'ifa_h':  (5.0*mm, 10.0*mm),
        'ifa_l':  (15*mm, 30*mm),
        'ifa_w1': (0.3*mm, 1.5*mm),
        'ifa_w2': (0.3*mm, 1.5*mm),
        'ifa_wf': (0.3*mm, 1.5*mm),
        'ifa_fp': (1*mm, 5*mm),
        'ifa_e2': (0.5*mm, 15*mm),
    }

    t0 = time.perf_counter()
    print("=== START ===", flush=True)

    best_params, result, summary = optimize_ifa(
        start_parameters=parameters,
        optimize_parameters=optimize_parameters,
        maxiter=50,
        popsize=10,
        seed=1,
        polish=False,
        # your bandwidth shaping:
        bandwidth_target_db=-10.0,
        bandwidth_span=(parameters['f1'], parameters['f2']),
        bandwidth_weight=2.0,
        # if you want every evaluation printed:
        # log_every_eval=True,
    )

    print("=== OPT DONE ===", flush=True)
    print("FINAL BEST PARAMS:", _fmt_params_singleline_mm(summary["best_params"], precision=3), flush=True)

    print("Success:", summary["optimizer_success"], flush=True)
    print("Message:", summary["optimizer_message"], flush=True)
    print("Objective value (=-RL-BWbonus):", summary["optimizer_fun"], flush=True)
    print("Best RL at f0 (dB):", summary["best_return_loss_dB_at_f0"], flush=True)
    print("=== END === elapsed: %.1fs" % (time.perf_counter()-t0), flush=True)

if __name__ == "__main__":  # crucial if you ever set workers != 1
    try:
        main()
    except Exception as e:
        print("FATAL ERROR:", e, file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
