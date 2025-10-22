from optimize_lib import optimize_ifa

mm = 1e-3
parameters = { 
    'ifa_h': 6.0*mm,
    'ifa_l': 26*mm,
    'ifa_w1': 0.619*mm,
    'ifa_w2': 0.5*mm,
    'ifa_wf': 0.5*mm,
    'ifa_fp': 2*mm,
    'ifa_e': 0.5*mm,
    'ifa_e2': 7*mm,
    'ifa_te': 0.5*mm,
    'via_size': 0.5*mm,
    'wsub': 21*mm,
    'hsub': 40*mm,
    'th': 1.5*mm,
    'mifa_meander': 2*mm,
    'mifa_meander_edge_distance': 3*mm,
    'mifa_tipdistance': 3*mm,
    'f1': 2.30e9,
    'f0': 2.45e9,
    'f2': 2.60e9,
    'freq_points': 3,
    'boundry_size_divisor': 0.4,
    'wavelength_fraction': 0.4,
}

# IMPORTANT: set bounds in METERS. Multiply EACH entry by mm.
optimize_parameters = { 
    'ifa_h':  (5.0*mm, 10.0*mm),
}

best_params, result, summary = optimize_ifa(
    start_parameters=parameters,
    optimize_parameters=optimize_parameters,
    maxiter=30,           # bump for better results (time ↑)
    popsize=14,           # population size per dim
    seed=1,
    polish=True,

    # Optional bandwidth shaping (uncomment if you want it)
    bandwidth_target_db=-10.0,
    bandwidth_span=(parameters['f1'], parameters['f2']),
    bandwidth_weight=2.0,   # reward wide -10 dB band
)

print("Success:", summary["optimizer_success"])
print("Message:", summary["optimizer_message"])
print("Objective value (=-RL-BWbonus):", summary["optimizer_fun"])
print("Best RL at f0 (dB):", summary["best_return_loss_dB_at_f0"])
print("Best parameters (m):")
for k, v in summary["best_params"].items():
    if k.endswith(('_h','_w','_l','_e','_wf','_fp','_te','_size','wsub','hsub','th')):
        print(f"  {k:>24s}: {v/mm:.3f} mm")
    else:
        print(f"  {k:>24s}: {v}")
