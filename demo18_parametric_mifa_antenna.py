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
    'wsub': 21*mm,# substrate width
    'hsub': 89.4*mm, # substrate length
    'th': 1.5*mm, # substrate thickness
    'mifa_meander': 2*mm,
    'mifa_meander_edge_distance': 2*mm,
    'mifa_tipdistance': 2*mm,
    'f1': 2.3e9,
    'f0': 2.45e9,
    'f2': 2.6e9,
    'freq_points': 3,
    'boundry_size_divisor': 0.33,
    'wavelength_fraction': 0.2,
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
    'wsub': 21*mm,# substrate width
    'hsub': 40*mm, # substrate length
    'th': 1.5*mm, # substrate thickness
    'mifa_meander': 2*mm,
    'mifa_meander_edge_distance':3*mm,
    'mifa_tipdistance':3*mm,
    'f1': 2.3e9,
    'f0': 2.45e9,
    'f2': 2.6e9,
    'freq_points': 3,
    'boundry_size_divisor': 0.4,
    'wavelength_fraction': 0.4,
}

mifa_test = { 'boundry_size_divisor': 0.33, 'f0': 2.45e+09, 'f1': 2.3e+09, 'f2': 2.6e+09, 'freq_points': 5, 'hsub': 90.000*mm, 'ifa_e': 0.500*mm, 'ifa_e2': 6.083*mm, 'ifa_fp': 4.436*mm, 'ifa_h': 9.268*mm, 'ifa_l': 24.927*mm, 'ifa_te': 0.500*mm, 'ifa_w1': 0.951*mm, 'ifa_w2': 0.842*mm, 'ifa_wf': 0.495*mm, 'mifa_meander': 2.000*mm, 'mifa_meander_edge_distance': 3.000*mm, 'mifa_tipdistance': 3.000*mm, 'th': 1.500*mm, 'via_size': 0.500*mm, 'wavelength_fraction': 0.2, 'wsub': 21.000*mm }
mifa_test = { 'boundry_size_divisor': 0.4, 'f0': 2.45e+09, 'f1': 2.3e+09, 'f2': 2.6e+09, 'freq_points': 3, 'hsub': 90.000*mm, 'ifa_e': 0.500*mm, 'ifa_e2': 0.889*mm, 'ifa_fp': 4.787*mm, 'ifa_h': 7.036*mm, 'ifa_l': 21.579*mm, 'ifa_te': 0.500*mm, 'ifa_w1': 0.815*mm, 'ifa_w2': 1.395*mm, 'ifa_wf': 0.425*mm, 'mifa_meander': 2.000*mm, 'mifa_meander_edge_distance': 3.000*mm, 'mifa_tipdistance': 3.000*mm, 'th': 1.500*mm, 'via_size': 0.500*mm, 'wavelength_fraction': 0.3, 'wsub': 21.000*mm }
#mifa_test = { 'boundry_size_divisor': 0.4, 'f0': 2.45e+09, 'f1': 2.45e+09, 'f2': 2.45e+09, 'freq_points': 3, 'hsub': 0.09, 'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_fp': 0.00454152959, 'ifa_h': 0.00720176584, 'ifa_l': 0.0211934952, 'ifa_te': 0.0005, 'ifa_w1': 0.00071611549, 'ifa_w2': 0.00140504788, 'ifa_wf': 0.000431540845, 'mifa_meander': 0.002, 'mifa_meander_edge_distance': 0.003, 'mifa_tipdistance': 0.003, 'th': 0.0015, 'via_size': 0.0005, 'wavelength_fraction': 0.3, 'wsub': 0.021 }
mifa_test = { 'boundry_size_divisor': 0.4, 'f0': 2.45e+09, 'f1': 2.2e+09, 'f2': 2.7e+09, 'freq_points': 3, 'hsub': 0.09, 'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_fp': 0.00454152959, 'ifa_h': 0.00720176584, 'ifa_l': 0.0211934952, 'ifa_te': 0.0005, 'ifa_w1': 0.00071611549, 'ifa_w2': 0.00140504788, 'ifa_wf': 0.000431540845, 'mifa_meander': 0.002, 'mifa_meander_edge_distance': 0.003, 'mifa_tipdistance': 0.003, 'th': 0.0015, 'via_size': 0.0005, 'wavelength_fraction': 0.3, 'wsub': 0.021 }
mifa_test = { 'boundry_size_divisor': 0.4, 'f0': 2.45e+09, 'f1': 2.3e+09, 'f2': 2.6e+09, 'freq_points': 3, 'hsub': 0.09, 'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_fp': 0.00459608786, 'ifa_h': 0.00743038264, 'ifa_l': 0.0211784476, 'ifa_te': 0.0005, 'ifa_w1': 0.000641190557, 'ifa_w2': 0.00138621391, 'ifa_wf': 0.000434786982, 'mifa_meander': 0.002, 'mifa_meander_edge_distance': 0.003, 'mifa_tipdistance': 0.003, 'th': 0.0015, 'via_size': 0.0005, 'wavelength_fraction': 0.3, 'wsub': 0.021, 'lambda_scale': 1 }
mifa_test =  { 'boundry_size_divisor': 0.4, 'f0': 2.45e+09, 'f1': 2.3e+09, 'f2': 2.6e+09, 'freq_points': 3, 'hsub': 0.09, 'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_fp': 0.00435061745, 'ifa_h': 0.00759614848, 'ifa_l': 0.0207929428, 'ifa_te': 0.0005, 'ifa_w1': 0.000542306047, 'ifa_w2': 0.00139626179, 'ifa_wf': 0.000441327827, 'mifa_meander': 0.002, 'mifa_meander_edge_distance': 0.003, 'mifa_tipdistance': 0.003, 'th': 0.0015, 'via_size': 0.0005, 'wavelength_fraction': 0.3, 'wsub': 0.021, 'lambda_scale': 1 }
 
parameters = mifa_test
parameters['boundry_size_divisor'] = 0.4
parameters['wavelength_fraction'] = 0.4
parameters['lambda_scale']=1

parameters['freq_points'] = 5
model, S11, freq_dense,ff1, ff2, ff3d = build_mifa(parameters,view_model=True,run_simulation=True,compute_farfield=False,loglevel="INFO",solver=em.EMSolver.CUDSS)

if S11 is not None:
    print(f"S11 at f0 frequency {parameters['f0'] / 1e9} GHz: {get_s11_at_freq(S11, parameters['f0'], freq_dense)} dB")
    print(f"S11 return loss (dB) at {parameters['f0']/1e9} GHz: {get_loss(S11, parameters['f0'], freq_dense)} dB")
    plot_sp(freq_dense, S11)                       # plot return loss in dB
