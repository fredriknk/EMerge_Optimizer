import emerge as em
import numpy as np
from ifalib import mm, get_s11_at_freq, get_loss_at_freq, get_resonant_frequency,get_bandwidth
from optimize_lib import _fmt_params_singleline_raw
from emerge.plot import plot_sp, smith, plot_ff_polar, plot_ff
from ifalib2 import AntennaParams, build_mifa, resolve_linked_params,unflatten_param_alias_dict



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
params = { 'p.board_wsub': 0.0191, 'p.board_th': 0.0015, 'p.sweep_freqs': np.array([2.0e+09, 2.2e+09, 2.9e+09, 3.0e+09]), 'p.sweep_weights': np.array([1., 1., 1., 1.]), 'p.board_hsub': 0.06, 'p.ifa_e': 0.0005, 'p.ifa_e2': 0.000575394784, 'p.ifa_fp': 0.00697749988, 'p.ifa_h': 0.0113544704, 'p.ifa_l': 0.0211655052, 'p.ifa_te': 0.0005, 'p.ifa_w1': 0.000461804774, 'p.ifa_w2': 0.000348731173, 'p.ifa_wf': 0.000304637534, 'p.mesh_boundary_size_divisor': 0.33, 'p.mesh_wavelength_fraction': 0.2, 'p.mifa_meander': 0.0023, 'p.mifa_low_dist': '${p.ifa_h} - 0.002', 'p.mifa_tipdistance': '${p.mifa_low_dist}', 'p.via_size': 0.0005, 'p.lambda_scale': 1, 'p2.ifa_l': 0.0180223593, 'p2.ifa_h': '${p.mifa_low_dist}- 0.0005', 'p2.ifa_e': '${p.ifa_fp}', 'p2.ifa_w2': 0.00044861946, 'p2.mifa_meander': '${p2.ifa_w2}*2+0.0003', 'p2.mifa_low_dist': '${p.mifa_low_dist} - 0.002', 'p2.mifa_tipdistance': '${p2.mifa_low_dist}', 'p2.shunt': 0 }
params = { 'p.board_wsub': 0.0191, 'p.board_th': 0.0015, 'p.sweep_freqs': np.array([2.0e+09, 2.2e+09, 2.9e+09, 3.0e+09]), 'p.sweep_weights': np.array([1., 1., 1., 1.]), 'p.board_hsub': 0.06, 'p.ifa_e': 0.0005, 'p.ifa_e2': 0.000575394784, 'p.ifa_fp': 0.00714452738, 'p.ifa_h': 0.0111977757, 'p.ifa_l': 0.0261440438, 'p.ifa_te': 0.0005, 'p.ifa_w1': 0.000439111941, 'p.ifa_w2': 0.000606869027, 'p.ifa_wf': 0.000320441544, 'p.mesh_boundary_size_divisor': 0.33, 'p.mesh_wavelength_fraction': 0.2, 'p.mifa_meander': 0.0023, 'p.mifa_low_dist': '${p.ifa_h} - 0.002', 'p.mifa_tipdistance': '${p.mifa_low_dist}', 'p.via_size': 0.0005, 'p.lambda_scale': 1, 'p2.ifa_l': 0.0247125508, 'p2.ifa_h': '${p.mifa_low_dist}- 0.0005', 'p2.ifa_e': '${p.ifa_fp}', 'p2.ifa_w2': 0.000420956532, 'p2.mifa_meander': '${p2.ifa_w2}*2+0.0003', 'p2.mifa_low_dist': '${p.mifa_low_dist} - 0.002', 'p2.mifa_tipdistance': '${p2.mifa_low_dist}', 'p2.shunt': 0 }
params = { 'p.board_wsub': 0.0191, 'p.board_th': 0.0015, 'p.sweep_freqs': np.linspace(1.9e9, 3.1e9, 11), 'p.sweep_weights': np.array([1., 1., 1., 1.]), 'p.board_hsub': 0.06, 'p.ifa_e': 0.0005, 'p.ifa_e2': 0.000575394784, 'p.ifa_fp': 0.00714452738, 'p.ifa_h': 0.0111977757, 'p.ifa_l': 0.0261440438, 'p.ifa_te': 0.0005, 'p.ifa_w1': 0.000439111941, 'p.ifa_w2': 0.000606869027, 'p.ifa_wf': 0.000320441544, 'p.mesh_boundary_size_divisor': 0.33, 'p.mesh_wavelength_fraction': 0.2, 'p.mifa_meander': 0.0023, 'p.mifa_low_dist': '${p.ifa_h} - 0.002', 'p.mifa_tipdistance': '${p.mifa_low_dist}', 'p.via_size': 0.0005, 'p.lambda_scale': 1, 'p2.ifa_l': 0.0247125508, 'p2.ifa_h': '${p.mifa_low_dist}- 0.0005', 'p2.ifa_e': '${p.ifa_fp}', 'p2.ifa_w2': 0.000420956532, 'p2.mifa_meander': '${p2.ifa_w2}*2+0.0003', 'p2.mifa_low_dist': '${p.mifa_low_dist} - 0.002', 'p2.mifa_tipdistance': '${p2.mifa_low_dist}', 'p2.shunt': 0 }
params = { 'p.board_wsub': 0.0191, 'p.board_th': 0.0015, 'p.sweep_freqs': np.array([2.3e+09, 2.4e+09, 2.9e+09, 3.0e+09]), 'p.sweep_weights': np.array([1., 1., 1., 1.]), 'p.board_hsub': 0.06, 'p.ifa_e': 0.0005, 'p.ifa_e2': 0.000575394784, 'p.ifa_fp': 0.0045, 'p.ifa_h': 0.008, 'p.ifa_l': 0.02, 'p.ifa_te': 0.0005, 'p.ifa_w1': 0.0005, 'p.ifa_w2': 0.001, 'p.ifa_wf': 0.0005, 'p.mesh_boundary_size_divisor': 0.33, 'p.mesh_wavelength_fraction': 0.2, 'p.mifa_meander': 0.0023, 'p.mifa_low_dist': '${p.ifa_h} - 0.003', 'p.mifa_tipdistance': '${p.mifa_low_dist}', 'p.via_size': 0.0005, 'p.lambda_scale': 1, 'p2.ifa_l': 0.021, 'p2.ifa_h': '${p.mifa_low_dist}- 0.0005', 'p2.ifa_e': '${p.ifa_fp}', 'p2.ifa_w2': 0.0006, 'p2.mifa_meander': '${p2.ifa_w2}*2+0.0003', 'p2.mifa_low_dist': '${p.mifa_low_dist} - 0.003', 'p2.mifa_tipdistance': '${p2.mifa_low_dist}', 'p2.shunt': 0 }
params = { 'p.board_wsub': 0.0191, 'p.board_th': 0.0015, 'p.sweep_freqs': np.linspace(1.9e9, 3.1e9, 11), 'p.sweep_weights': np.array([1., 1., 1., 1.]), 'p.board_hsub': 0.06, 'p.ifa_e': 0.0005, 'p.ifa_e2': 0.000575394784, 'p.ifa_fp': 0.00492386667, 'p.ifa_h': 0.0112790182, 'p.ifa_l': 0.03313721, 'p.ifa_te': 0.0005, 'p.ifa_w1': 0.00117204977, 'p.ifa_w2': 0.000471475221, 'p.ifa_wf': 0.000876969117, 'p.mesh_boundary_size_divisor': 0.33, 'p.mesh_wavelength_fraction': 0.2, 'p.mifa_meander': 0.0023, 'p.mifa_low_dist': '${p.ifa_h} - 0.003', 'p.mifa_tipdistance': '${p.mifa_low_dist}', 'p.via_size': 0.0005, 'p.lambda_scale': 1, 'p2.ifa_l': 0.0292781434, 'p2.ifa_h': '${p.mifa_low_dist}- 0.0005', 'p2.ifa_e': '${p.ifa_fp}', 'p2.ifa_w2': 0.000740958143, 'p2.mifa_meander': '${p2.ifa_w2}*2+0.0003', 'p2.mifa_low_dist': '${p.mifa_low_dist} - 0.003', 'p2.mifa_tipdistance': '${p2.mifa_low_dist}', 'p2.shunt': 0 }
params = { 'p.board_wsub': 0.0191, 'p.board_th': 0.0015, 'p.sweep_freqs': np.linspace(1.9e9, 5.1e9, 11), 'p.board_hsub': 0.06, 'p.ifa_e': 0.0005, 'p.ifa_e2': 0.000575394784, 'p.ifa_fp': 0.0103088506, 'p.ifa_h': 0.0100941112, 'p.ifa_l': 0.00666845158, 'p.ifa_te': 0.0005, 'p.ifa_w1': 0.000674611784, 'p.ifa_w2': 0.000429060267, 'p.ifa_wf': 0.000423134131, 'p.mesh_boundary_size_divisor': 0.33, 'p.mesh_wavelength_fraction': 0.2, 'p.mifa_meander': 0.0023, 'p.mifa_low_dist': '${p.ifa_h} - 0.003', 'p.mifa_tipdistance': '${p.mifa_low_dist}', 'p.via_size': 0.0005, 'p.lambda_scale': 1, 'p2.ifa_l': 0.0336236949, 'p2.ifa_h': '${p.mifa_low_dist}- 0.0005', 'p2.ifa_e': '${p.ifa_fp}', 'p2.ifa_w2': 0.000493602078, 'p2.mifa_meander': '${p2.ifa_w2}*2+0.0003', 'p2.mifa_low_dist': '${p.mifa_low_dist} - 0.003', 'p2.mifa_tipdistance': '${p2.mifa_low_dist}', 'p2.shunt': 0 }
params = { 'p.board_wsub': 0.0191, 'p.board_th': 0.0015, 'p.sweep_freqs': np.concatenate((np.linspace(2.0e9, 2.8e9, 5),np.linspace(4.7e9, 5.5e9, 5))), 'p.board_hsub': 0.06, 'p.ifa_e': 0.0005, 'p.ifa_e2': 0.000575394784, 'p.ifa_fp': 0.00364461081, 'p.ifa_h': 0.00909984885, 'p.ifa_l': 0.0355663827, 'p.ifa_te': 0.0005, 'p.ifa_w1': 0.00112657281, 'p.ifa_w2': 0.000445781771, 'p.ifa_wf': 0.000398836163, 'p.mesh_boundary_size_divisor': 0.33, 'p.mesh_wavelength_fraction': 0.2, 'p.mifa_meander': 0.0023, 'p.mifa_low_dist': '${p.ifa_h} - 0.003', 'p.mifa_tipdistance': '${p.mifa_low_dist}', 'p.via_size': 0.0005, 'p.lambda_scale': 1, 'p2.ifa_l': 0.00796519359, 'p2.ifa_h': '${p.mifa_low_dist}- 0.0005', 'p2.ifa_e': '${p.ifa_fp}', 'p2.ifa_w2': 0.00033083229, 'p2.mifa_meander': '${p2.ifa_w2}*2+0.0003', 'p2.mifa_low_dist': '${p.mifa_low_dist} - 0.003', 'p2.mifa_tipdistance': '${p2.mifa_low_dist}', 'p2.shunt': 0 }

params = unflatten_param_alias_dict(params)

parameters = params

if __name__=="__main__":
    
    model, S11, freq_dense,ff1, ff2, ff3d = build_mifa(parameters,
                                                   view_mesh=False, view_model=True,
                                                   run_simulation=True,compute_farfield=True,
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