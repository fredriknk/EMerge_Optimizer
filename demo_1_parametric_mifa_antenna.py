import emerge as em
import numpy as np
from ifalib2 import build_mifa, get_s11_at_freq, get_loss_at_freq, get_resonant_frequency,get_bandwidth
#from ifalib2 import AntennaParams, build_mifa
from optimize_lib import _fmt_params_singleline_raw
from emerge.plot import plot_sp, smith, plot_ff_polar, plot_ff
from numpy import array

""" PATCH ANTENNA DEMO

This design is modeled after this Comsol Demo: https://www.comsol.com/model/microstrip-patch-antenna-11742

In this demo we build and simulate a rectangular patch antenna on a dielectric
substrate with airbox and lumped port excitation, then visualize S-parameters
and far-field radiation patterns. 

This simulation is quite heavy and might take a while to fully compute.
"""
# --- Unit and simulation parameters --------------------------------------
mm = 0.001              # meters per millimeter

# --- Antenna geometry dimensions ----------------------------------------

mifa_21x90_2450mhz = {
    'board_wsub': 0.021, 
    'board_th': 0.0015,
    'f0': 2.45e+09, 
    'f1': 2.3e+09, 
    'f2': 2.6e+09, 
    'freq_points': 5, 
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
    'mesh_boundary_size_divisor': 0.33,
    'mesh_wavelength_fraction': 0.2, 
    'mifa_meander': 0.002, 
    'mifa_low_dist': 0.003, 
    'mifa_tipdistance': 0.003, 
    'via_size': 0.0005,  
    'eps_r': 3.38,
    'lambda_scale': 1 }

mifa_14x25_2450mhz = { 
    'ifa_h': 0.00773189309, 'ifa_l': 0.0229509148, 
    'ifa_w1': 0.000766584703, 'ifa_w2': 0.000440876843, 'ifa_wf': 0.000344665757, 
    'ifa_fp': 0.00156817497, 'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_te': 0.0005, 
    'via_size': 0.0003, 'board_wsub': 0.014, 'board_hsub': 0.025, 'board_th': 0.0015, 
    'mifa_meander': 0.00195527223, 'mifa_low_dist': 0.00217823618, 
    'f1': 2.3e+09, 'f0': 2.45e+09, 'f2': 2.7e+09, 'freq_points': 5, 'eps_r': 3.38,
    'mesh_boundary_size_divisor': 0.33, 'mesh_wavelength_fraction': 0.2, 'lambda_scale': 1 }

mifa_30x110_821mhz = { 
    'ifa_h': 0.0230161676, 'ifa_l': 0.107128555, 
    'ifa_w1': 0.00045360957, 'ifa_w2': 0.000439352308, 'ifa_wf': 0.000411214503, 
    'ifa_fp': 0.00722339282, 'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_te': 0.0005, 
    'via_size': 0.0005, 'board_wsub': 0.03, 'board_hsub': 0.11, 'board_th': 0.0015, 
    'mifa_meander': 0.00169312729, 'mifa_low_dist': 0.003, 'mifa_tipdistance': 0.003,
    'f1': 791000000, 'f0': 826000000, 'f2': 862000000, 'freq_points': 3, 'eps_r': 3.38,
    'mesh_boundary_size_divisor': 0.33, 'mesh_wavelength_fraction': 0.2, 'lambda_scale': 1, 
    }


parameters = mifa_14x25_2450mhz

parameters['f1'] = parameters['f1'] - 1e8
parameters['f2'] = parameters['f2'] + 1e8
parameters['freq_points'] = 5

model, S11, freq_dense,ff1, ff2, ff3d = build_mifa(parameters,
                                                   view_mesh=True, view_model=True,
                                                   run_simulation=True,compute_farfield=True,
                                                   loglevel="INFO",solver=em.EMSolver.CUDSS)

if S11 is not None:
    print(_fmt_params_singleline_raw(parameters))
    RL_dB = -20*np.log10(np.abs(S11))
    idx_min = np.argmax(RL_dB)
    f_resonant = freq_dense[idx_min]

    print(f"idx_min: {idx_min}, rl_min: {RL_dB[idx_min]:.2f} dB at f_resonant: {f_resonant/1e9:.4f} GHz")
    print(f"S11 at f0 frequency {parameters['f0'] / 1e9} GHz: {get_s11_at_freq(S11, parameters['f0'], freq_dense)} dB")
    print(f"S11 return loss (dB) at {parameters['f0']/1e9} GHz: {get_loss_at_freq(S11, parameters['f0'], freq_dense)} dB")
    print(f"Resonant frequency (min |S11|): {get_resonant_frequency(S11, freq_dense)/1e9} GHz")
    bw = get_bandwidth(S11, freq_dense, rl_threshold_dB=-10, f0=parameters['f0'])
    print(f"Bandwidth (-10 dB): {(bw[1]-bw[0])/1e6} MHz, from/to {bw/1e6} MHz")
    
    plot_sp(freq_dense, S11)                       # plot return loss in dB
    smith(S11, f=freq_dense, labels='S11')         # Smith chart of S11

    
    # --- Far-field radiation pattern ----------------------------------------
    # Extract 2D cut at phi=0 plane and plot E-field magnitude

if ff1 is not None and ff2 is not None:
    plot_ff(ff1.ang*180/np.pi, [ff1.normE/em.lib.EISO, ff2.normE/em.lib.EISO], dB=True, ylabel='Gain [dBi]')                # linear plot vs theta
    plot_ff_polar(ff1.ang, [ff1.normE/em.lib.EISO, ff2.normE/em.lib.EISO], dB=True, dBfloor=-20)          # polar plot of radiation
    surf = ff3d.surfplot('normE', rmax=60 * mm)
    model.display.add_surf(*surf)
    model.display.show()
