import emerge as em
import numpy as np
from ifalib2 import mm, get_s11_at_freq, get_loss_at_freq, get_resonant_frequency,get_bandwidth
from optimize_lib import _fmt_params_singleline_raw
from emerge.plot import plot_sp, smith, plot_ff_polar, plot_ff
from ifalib2 import AntennaParams, build_mifa, resolve_linked_params
from numpy import array

mifa_2450_5800mhz = { #Multifrequency MIFA from optimized with global optimizer
    #Main antenna parameters
    'p.board_wsub': 0.0191, 'p.board_th': 0.0015, 'p.board_hsub': 0.06,
    'p.sweep_freqs': array([2.45e+09, 5.80e+09]), 
    'p.sweep_weights': array([1., 1.]), 
    'p.ifa_e': 0.0005, 'p.ifa_e2': 0.000575394784, 'p.ifa_fp': 0.00272626454, 
    'p.ifa_h': 0.00829488465, 'p.ifa_l': 0.0234122232, 'p.ifa_te': 0.0005,
    'p.ifa_w1': 0.000435609747, 'p.ifa_w2': 0.000537390996, 'p.ifa_wf': 0.00062670221, 
    'p.mesh_boundary_size_divisor': 0.33, 'p.mesh_wavelength_fraction': 0.2,'p.lambda_scale': 1,
    'p.mifa_meander': 0.0023, 
    'p.mifa_low_dist': '${p.ifa_h} - 0.003', #Parametric link to ifa_h to simplify optimization
    'p.mifa_tipdistance': '${p.mifa_low_dist}', 
    'p.via_size': 0.0005,
    
    #Secondary antenna stub parameters, as long as we dont assign a new fp, it will have the 
    #same origin
    'p2.ifa_l': 0.00601520693, 
    'p2.ifa_h': '${p.mifa_low_dist}- 0.0005', 
    'p2.ifa_e': '${p.ifa_fp}', # we set ifa_e to ifa_fp since we don't use shunt 
    'p2.ifa_w2': 0.000388412941, 
    'p2.mifa_meander': '${p2.ifa_w2}*2+0.0003',#Parametric links can have arithmetic expressions 
    'p2.mifa_low_dist': '${p.mifa_low_dist} - 0.003', 
    'p2.mifa_tipdistance': '${p2.mifa_low_dist}', 
    'p2.shunt': False #Shunt is turned off on second antenna stub (Can be defined, but then set p2.ifa_e accordingly)
    #All other parameters use default values from p.X
    } 

params = { 'p.board_wsub': 0.0191, 'p.board_hsub': 0.06, 'p.board_th': 0.0015, 'p.lambda_scale': 1, 'p.sweep_freqs': array([2.45e+09, 5.80e+09]), 'p.sweep_weights': array([1., 1.]), 'p.ifa_l': 0.0238644296, 'p.ifa_h': 0.00785785202, 'p.ifa_w1': 0.000364937869, 'p.ifa_w2': 0.00045136907, 'p.ifa_wf': 0.00062670221, 'p.ifa_fp': 0.00272626454, 'p.ifa_e': 0.0005, 'p.ifa_e2': 0.000575394784, 'p.ifa_te': 0.0005, 'p.mifa_meander': 0.0023, 'p.mifa_low_dist': 0.00529488465, 'p.mifa_tipdistance': 0.00529488465, 'p.via_size': 0.0005, 'p.shunt': 1, 'p.mesh_boundary_size_divisor': 0.33, 'p.mesh_wavelength_fraction': 0.2, 'p.eps_r': 4.4, 'p.validate': 1, 'p.clearance': 0.0003, 'p2.ifa_l': 0.00601520693, 'p2.ifa_h': 0.00479488465, 'p2.ifa_w2': 0.000388412941, 'p2.ifa_e': 0.00272626454, 'p2.mifa_meander': 0.00107682588, 'p2.mifa_low_dist': 0.00229488465, 'p2.mifa_tipdistance': 0.00229488465, 'p2.shunt': 0 }
parameters = params

if __name__=="__main__":
    p = parameters.copy()
    print(f"Main Antenna: {_fmt_params_singleline_raw(p)}")
    
    model, S11, freq_dense,ff1, ff2, ff3d = build_mifa(p,
                                                   view_mesh=False, view_model=True,
                                                   run_simulation=True,compute_farfield=False,
                                                   loglevel="INFO",solver=em.EMSolver.CUDSS,)
    
    if S11 is not None:
        

        
        RL_dB = -20*np.log10(np.abs(S11))
        idx_min = np.argmax(RL_dB)
        f_resonant = freq_dense[idx_min]

        print(f"idx_min: {idx_min}, rl_min: {RL_dB[idx_min]:.2f} dB at f_resonant: {f_resonant/1e9:.4f} GHz")
        print(f"S11 return loss (dB) at {p['p.sweep_freqs']/1e9} GHz: {get_loss_at_freq(S11, p['p.sweep_freqs'], freq_dense)} dB")
        print(f"Resonant frequency (min |S11|): {get_resonant_frequency(S11, freq_dense)/1e9} GHz")
        
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