import emerge as em
import numpy as np
from ifalib import mm, get_s11_at_freq, get_loss_at_freq, get_resonant_frequency,get_bandwidth
from optimize_lib import _fmt_params_singleline_raw
from emerge.plot import plot_sp, smith, plot_ff_polar, plot_ff
from ifalib2 import AntennaParams, build_mifa, _resolve_linked_params



params ={
    "p": {
        'board_wsub': 0.021, 
        'board_th': 0.0015,
        'f1': 2.0e+09, 
        'f2': 3.0e+09, 
        'f0': 2.0e+09,
        'freq_points': 2, 
        'board_hsub': 0.09, 
        'ifa_e': 0.0005, 
        'ifa_e2': 0.000575394784, 
        'ifa_fp': 0.00378423695, 
        'ifa_h': 0.00790411482, 
        'ifa_l': 0.020, 
        'ifa_te': 0.0005, 
        'ifa_w1': 0.000550173526, 
        'ifa_w2': 0.001, 
        'ifa_wf': 0.000433478781, 
        'mesh_boundary_size_divisor': 0.33,
        'mesh_wavelength_fraction': 0.2, 
        'mifa_meander': 0.001*2+0.0003, 
        'mifa_low_dist': "${p.ifa_h} - 0.002", 
        'mifa_tipdistance': "${p.mifa_low_dist}", 
        'via_size': 0.0005,  
        'lambda_scale': 1 
    },
    "p2" : {
        "f0":3.0e+09,
        "ifa_l":0.019,
        "ifa_h":"${p.mifa_low_dist}- 0.0005",
        "ifa_e":"${p.ifa_fp}",
        "ifa_w2":0.0005,
        "mifa_meander":"${p2.ifa_w2}*2+0.0003",
        "mifa_low_dist":"${p.mifa_low_dist} - 0.002",
        "mifa_tipdistance":"${p2.mifa_low_dist}",
        "shunt":False,
    },
}

parameters = params

if __name__=="__main__":
    
    model, S11, freq_dense,ff1, ff2, ff3d = build_mifa(parameters,
                                                   view_mesh=False, view_model=False,
                                                   run_simulation=True,compute_farfield=False,
                                                   loglevel="INFO",solver=em.EMSolver.CUDSS)
    
    
    if S11 is not None:
        print(f"Main Antenna: {_fmt_params_singleline_raw(parameters)}")
        p = parameters['p']
        p2 = parameters['p2']
        RL_dB = -20*np.log10(np.abs(S11))
        idx_min = np.argmax(RL_dB)
        f_resonant = freq_dense[idx_min]

        print(f"idx_min: {idx_min}, rl_min: {RL_dB[idx_min]:.2f} dB at f_resonant: {f_resonant/1e9:.4f} GHz")
        print(f"S11 at f0 frequency {p['f0'] / 1e9} GHz: {get_s11_at_freq(S11, p['f0'], freq_dense)} dB")
        print(f"S11 return loss (dB) at {p['f0']/1e9} GHz: {get_loss_at_freq(S11, p['f0'], freq_dense)} dB")
        print(f"S11 at f0 frequency {p2['f0'] / 1e9} GHz: {get_s11_at_freq(S11, p['f0'], freq_dense)} dB")
        print(f"S11 return loss (dB) at {p2['f0']/1e9} GHz: {get_loss_at_freq(S11, p2['f0'], freq_dense)} dB")
        print(f"Resonant frequency (min |S11|): {get_resonant_frequency(S11, freq_dense)/1e9} GHz")
        #bw = get_bandwidth(S11, freq_dense, rl_threshold_dB=-10, f0=p['f0'])
        #print(f"Bandwidth (-10 dB): {(bw[1]-bw[0])/1e6} MHz, from/to {bw/1e6} MHz")
        
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