import emerge as em
import numpy as np
from ifalib import build_mifa, get_s11_at_freq, get_loss
from emerge.plot import plot_sp, smith, plot_ff_polar, plot_ff

""" PATCH ANTENNA DEMO

This design is modeled after this Comsol Demo: https://www.comsol.com/model/microstrip-patch-antenna-11742

In this demo we build and simulate a rectangular patch antenna on a dielectric
substrate with airbox and lumped port excitation, then visualize S-parameters
and far-field radiation patterns. 

This simulation is quite heavy and might take a while to fully compute.

#############################################################
#|------------- substrate_width -------------------|
# _______________________________________________     _ substrate_thickness
#| A  ifa_e      |----------ifa_l(total length)-| |\  \-gndplane_position 
#| V____          _______________     __________  | |  \_0 point
#|               |    ___  ___   |___|  ______  | | |
#|         ifa_h |   |   ||   |_________|    |  |_|_|_ mifa_meander_edge_distance 
#|               |   |   ||  mifa_meander    |__|_|_|_ mifa_tipdistance
#|               |   |   ||                   w2  | | |                  
#|_______________|___|___||_______________________| |_|
#| <---ifa_e---->| w1|   wf\                      | |
#|               |__fp___|  \                     | |
#|                       |    feed point          | |
#|                       |                        | | substrate_length
#|<- substrate_width/2 ->|                        | |
#|                                                | |
#|________________________________________________| |
# \________________________________________________\|
#############################################################
Note: ifa_l is total length including meanders and tip
"""
# --- Unit and simulation parameters --------------------------------------
mm = 0.001              # meters per millimeter

# --- Antenna geometry dimensions ----------------------------------------

ifa = {
    'ifa_h': 6.0*mm,
    'ifa_l': 19.09*mm,
    'ifa_w1': 0.7*mm,
    'ifa_w2': 0.705*mm,
    'ifa_wf': 0.7*mm,
    'ifa_fp': 2*mm,
    'ifa_e': 0.5*mm,
    'ifa_e2': 0.5*mm,
    'ifa_te': 0.5*mm,
    'via_size': 0.5*mm,
    'board_wsub': 21*mm,# substrate width
    'board_hsub': 89.4*mm, # substrate length
    'board_th': 1.5*mm, # substrate thickness
    'mifa_meander': 2*mm,
    'mifa_meander_edge_distance': 2*mm,
    'mifa_tipdistance': 2*mm,
    'f1': 2.3e9,
    'f0': 2.45e9,
    'f2': 2.6e9,
    'freq_points': 3,
    'mesh_boundry_size_divisor': 0.33,
    'mesh_wavelength_fraction': 0.2,
}

mifa = {
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
    'board_wsub': 21*mm,# substrate width
    'board_hsub': 40*mm, # substrate length
    'board_th': 1.5*mm, # substrate thickness
    'mifa_meander': 2*mm,
    'mifa_meander_edge_distance':3*mm,
    'mifa_tipdistance':3*mm,
    'f1': 2.3e9,
    'f0': 2.45e9,
    'f2': 2.6e9,
    'freq_points': 3,
    'mesh_boundry_size_divisor': 0.4,
    'mesh_wavelength_fraction': 0.4,
}

mifa_21x90_2_45ghz = {
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

test = { 'ifa_h': 0.0296763865, 'ifa_l': 0.130751561, 'ifa_w1': 0.000756940097, 'ifa_w2': 0.000848497577, 'ifa_wf': 0.00123205059, 'ifa_fp': 0.00905771941, 'ifa_e': 0.0005, 'ifa_e2': 0.00285990861, 'ifa_te': 0.0005, 'via_size': 0.0005, 'board_wsub': 0.03, 'board_hsub': 0.11, 'board_th': 0.0015, 'mifa_meander': 0.002, 'mifa_meander_edge_distance': 0.003, 'mifa_tipdistance': 0.003, 'f1': 700000000, 'f0': 800000000, 'f2': 900000000, 'freq_points': 3, 'mesh_boundry_size_divisor': 0.5, 'mesh_wavelength_fraction': 0.5, 'lambda_scale': 0.5 }

parameters = test

# parameters['mesh_boundry_size_divisor'] = 0.33
# parameters['mesh_wavelength_fraction'] = 0.2
# parameters['lambda_scale']=1
# parameters['freq_points'] = 5

model, S11, freq_dense,ff1, ff2, ff3d = build_mifa(parameters,view_mesh=True, view_model=True,run_simulation=True,compute_farfield=False,loglevel="INFO",solver=em.EMSolver.CUDSS)

if S11 is not None:
    print(f"S11 at f0 frequency {parameters['f0'] / 1e9} GHz: {get_s11_at_freq(S11, parameters['f0'], freq_dense)} dB")
    print(f"S11 return loss (dB) at {parameters['f0']/1e9} GHz: {get_loss(S11, parameters['f0'], freq_dense)} dB")
    plot_sp(freq_dense, S11)                       # plot return loss in dB
