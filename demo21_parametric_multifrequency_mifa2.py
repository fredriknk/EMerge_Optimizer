import emerge as em
import numpy as np
from ifalib import mm, get_s11_at_freq, get_loss_at_freq, get_resonant_frequency,get_bandwidth
from optimize_lib import _fmt_params_singleline_raw
from emerge.plot import plot_sp, smith, plot_ff_polar, plot_ff
from ifalib2 import AntennaParams, build_mifa, resolve_linked_params



params ={
    "p": {
        'board_wsub': 0.0191, 
        'board_th': 0.0015,
        'sweep_freqs': np.array([2.9e9,2.95e9,3.0e9]),
        'sweep_weights': np.array([1.0,1.0,1.0]),
        'freq_points': 10, 
        'board_hsub': 0.09, 
        'ifa_e': 0.0005, 
        'ifa_e2': 0.000575394784, 
        'ifa_fp': 0.0045, 
        'ifa_h': 0.00790411482, 
        'ifa_l': 0.020, 
        'ifa_te': 0.0005, 
        'ifa_w1': 0.0005, 
        'ifa_w2': 0.001, 
        'ifa_wf': 0.0005, 
        'mesh_boundary_size_divisor': 0.33,
        'mesh_wavelength_fraction': 0.2, 
        'mifa_meander': 0.001*2+0.0003, 
        'mifa_low_dist': "${p.ifa_h} - 0.002", 
        'mifa_tipdistance': "${p.mifa_low_dist}", 
        'via_size': 0.0005,  
        'lambda_scale': 1 
    },
    "p2" : {
        "ifa_l":0.021,
        "ifa_h":"${p.mifa_low_dist}- 0.0005",
        "ifa_e":"${p.ifa_fp}",
        "ifa_w2":0.0006,
        "mifa_meander":"${p2.ifa_w2}*2+0.0003",
        "mifa_low_dist":"${p.mifa_low_dist} - 0.002",
        "mifa_tipdistance":"${p2.mifa_low_dist}",
        "shunt":False,
    },
}

params= { 
    'p.board_wsub': 0.0191,
    'p.board_th': 0.0015, 
    'p.board_hsub': 0.06,
    'p.sweep_freqs': np.array([2.0e+09, 2.2e+09, 2.9e+09, 3.0e+09]), 
    'p.sweep_weights': np.array([1., 1., 1., 1.]), 
    'p.ifa_e': 0.0005, 
    'p.ifa_e2': 0.000575394784, 
    'p.ifa_fp': 0.00714452738, 
    'p.ifa_h': 0.0111977757, 
    'p.ifa_l': 0.0261440438, 
    'p.ifa_te': 0.0005, 
    'p.ifa_w1': 0.000439111941,
    'p.ifa_w2': 0.000606869027, 
    'p.ifa_wf': 0.000320441544, 
    'p.mesh_boundary_size_divisor': 0.33, 
    'p.mesh_wavelength_fraction': 0.2,
    'p.mifa_meander': 0.0023, 
    'p.mifa_low_dist': '${p.ifa_h} - 0.002', 
    'p.mifa_tipdistance': '${p.mifa_low_dist}', 
    'p.via_size': 0.0005, 
    'p.lambda_scale': 1, 
    
    'p2.ifa_l': 0.0247125508, 
    'p2.ifa_h': '${p.mifa_low_dist}- 0.0005', 
    'p2.ifa_e': '${p.ifa_fp}', 'p2.ifa_w2': 0.000420956532, 
    'p2.mifa_meander': '${p2.ifa_w2}*2+0.0003', 
    'p2.mifa_low_dist': '${p.mifa_low_dist} - 0.002', 
    'p2.mifa_tipdistance': '${p2.mifa_low_dist}', 
    'p2.shunt': 0 
}

parameters = params

if __name__=="__main__":
    
    model, S11, freq_dense,ff1, ff2, ff3d = build_mifa(parameters,
                                                   view_mesh=False, view_model=False,
                                                   run_simulation=True,compute_farfield=False,
                                                   loglevel="INFO",solver=em.EMSolver.CUDSS,)
    
    if S11 is not None:
        print(f"Main Antenna: {_fmt_params_singleline_raw(parameters)}")
        p = parameters['p']
        p2 = parameters['p2']
        RL_dB = -20*np.log10(np.abs(S11))
        idx_min = np.argmax(RL_dB)
        f_resonant = freq_dense[idx_min]

        print(f"idx_min: {idx_min}, rl_min: {RL_dB[idx_min]:.2f} dB at f_resonant: {f_resonant/1e9:.4f} GHz")
        print(f"S11 return loss (dB) at {p['sweep_freqs']/1e9} GHz: {get_loss_at_freq(S11, p['sweep_freqs'], freq_dense)} dB")
        print(f"S11 return loss (dB) at {p['sweep_freqs']/1e9} GHz: {get_loss_at_freq(S11, p['sweep_freqs'], freq_dense)} dB")
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