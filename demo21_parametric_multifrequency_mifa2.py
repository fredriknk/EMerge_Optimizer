import emerge as em
import numpy as np
from ifalib import mm, get_s11_at_freq, get_loss_at_freq, get_resonant_frequency,get_bandwidth
from optimize_lib import _fmt_params_singleline_raw
from emerge.plot import plot_sp, smith, plot_ff_polar, plot_ff
from ifalib2 import AntennaParams, build_mifa, resolve_linked_params

params= { 'p.board_wsub': 0.0191, 
         'p.board_th': 0.0015, 
         'p.sweep_freqs': np.array([2.45e+09, 5.00e+09]), 
         'p.sweep_weights': np.array([1., 1.]), 
         'p.board_hsub': 0.06, 'p.ifa_e': 0.0005, 
         'p.ifa_e2': 0.000575394784, 'p.ifa_fp': 0.00364461081, 
         'p.ifa_h': 0.00909984885, 'p.ifa_l': 0.0355663827, 
         'p.ifa_te': 0.0005, 
         'p.ifa_w1': 0.00112657281, 'p.ifa_w2': 0.000445781771, 'p.ifa_wf': 0.000398836163, 
         'p.mesh_boundary_size_divisor': 0.33, 
         'p.mesh_wavelength_fraction': 0.2, 'p.mifa_meander': 0.0023, 
         'p.mifa_low_dist': '${p.ifa_h} - 0.003', 
         'p.mifa_tipdistance': '${p.mifa_low_dist}', 
         'p.via_size': 0.0005, 
         'p.lambda_scale': 1, 
         
         'p2.ifa_l': 0.00796519359, 'p2.ifa_h': '${p.mifa_low_dist}- 0.0005', 
         'p2.ifa_e': '${p.ifa_fp}', # When using no shunt set ifa_e to ifa_fp to align
         'p2.ifa_w2': 0.00033083229, 
         'p2.mifa_meander': '${p2.ifa_w2}*2+0.0003', 
         'p2.mifa_low_dist': '${p.mifa_low_dist} - 0.003', 
         'p2.mifa_tipdistance': '${p2.mifa_low_dist}', 
         'p2.shunt': 0 }


parameters = params

if __name__=="__main__":
    
    model, S11, freq_dense,ff1, ff2, ff3d = build_mifa(parameters,
                                                   view_mesh=False, view_model=True,
                                                   run_simulation=True,compute_farfield=False,
                                                   loglevel="INFO",solver=em.EMSolver.CUDSS,)
    
    if S11 is not None:
        
        p = parameters
        
        p['p.lambda_scale'] = 1  # Ensure scale is 1 for frequency calculations
        p['p.mesh_wavelength_fraction'] = 0.2
        p['p.mesh_boundary_size_divisor'] = 0.33
        p['p.sweep_freqs'] = np.linspace(p['p.sweep_freqs'][0], p['p.sweep_freqs'][-1], 11)
        
        print(f"Main Antenna: {_fmt_params_singleline_raw(p)}")
        
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