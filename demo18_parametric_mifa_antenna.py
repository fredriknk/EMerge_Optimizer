import emerge as em
import numpy as np
import ifalib
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
    'f2': 2.6e9,
    'freq_points': 3,
    'boundry_size_divisor': 0.4,
    'wavelength_fraction': 0.4,
}

parameters = ifa

model, S11, freq_dense,ff1, ff2, ff3d = ifalib.build_mifa(parameters,view_model=False,run_simulation=True,compute_farfield=False)



plot_sp(freq_dense, S11)                       # plot return loss in dB
smith(S11, f=freq_dense, labels='S11')         # Smith chart of S11


if ff1 is not None:
    # reflection coefficient

    plot_ff(ff1.ang*180/np.pi, [ff1.normE/em.lib.EISO, ff2.normE/em.lib.EISO], dB=True, ylabel='Gain [dBi]')                # linear plot vs theta
    plot_ff_polar(ff1.ang, [ff1.normE/em.lib.EISO, ff2.normE/em.lib.EISO], dB=True, dBfloor=-20)          # polar plot of radiation

    # --- 3D radiation visualization -----------------------------------------
    # Add geometry to 3D display

    surf = ff3d.surfplot('normE', rmax=60 * mm,)

    model.display.add_surf(*surf)
    model.display.show()