from pyexpat import model
import emerge as em
import numpy as np
from ifa_validation import validate_ifa_params

mm = 0.001  # meters per millimeter

def build_mifa_plates(
    parameters = None,
    # --- placement ---
    tl=np.array([0.0, 0.0, 0.0]),  # global translation vector
    # --- bookkeeping ---
    name_prefix="ifa",
    priority=10,
):
    """
    Returns a list of em.geo.XYPlate objects representing the IFA radiator geometry
    (no meshing calls). Coordinates follow your ASCII sketch; z=0 plane.
    """
    if parameters is None:
        raise ValueError("parameters dict is required.")
    
    # --- board / antenna params (all in meters) ---
    substrate_width = parameters['board_wsub']          # total PCB width (x extent available for the radiator)
    
    ifa_l = parameters['ifa_l']                    # total desired electrical length (projected along x)
    ifa_h = parameters['ifa_h']                    # height of radiator above ground (y)
    ifa_w2 = parameters['ifa_w2']                   # radiator strip width
    ifa_fp = parameters['ifa_fp']                   # feed offset from board origin along x (see your sketch)
    ifa_e = parameters['ifa_e']                    # edge clearance along x (keep-out from left)
    ifa_e2 = parameters['ifa_e2']                   # edge clearance along x (keep-out from right)
    
    # --- meander/tip geometry knobs ---
    mifa_meander = parameters['mifa_meander']             # horizontal meander step (x)
    mifa_meander_edge_distance = parameters['mifa_meander_edge_distance']  # y-clearance from ground edge for meanders
    mifa_tipdistance = parameters.get('mifa_tipdistance', mifa_meander_edge_distance)         # y-clearance for the tip element

    mm = 0.001  # meters per millimeter
    plates = []

    def add_box_from_start_stop(start, stop, tag):
        """Convert 'start/stop' corners into an XYPlate and append it."""
        x0, y0, _ = start
        x1, y1, _ = stop
        z = start[2]
        pos = (min(x0, x1), min(y0, y1), z)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        if w <= 0 or h <= 0:
            return None
        plate = em.geo.XYPlate(
            width=w, depth=h, position=pos, name=f"{name_prefix}_{tag}"
        )
        plates.append(plate)
        return plate

    # Helper to add the horizontal base after tip/meanders are placed
    def add_base():
        add_box_from_start_stop(start_main, stop_main, "base")

    # --- Radiating element root (horizontal run) ---
    # Left/lower reference per your drawing:
    # start = [-ifa_fp, +ifa_h] + tl, stop = start + [length, -ifa_w2]
    start_main = np.array([-ifa_fp+ifa_e, ifa_h, 0.0]) + tl
    usable_x = substrate_width - ifa_e - ifa_e2
    length_diff = 0.0

    if ifa_l <= usable_x:
        # Simple straight radiator
        stop_main = start_main + np.array([ifa_l, -ifa_w2, 0.0])
        add_box_from_start_stop(start_main, stop_main, "main")
        return plates  # done

    # We need tip (vertical) + potential meanders
    start_main = np.array([-ifa_fp+ifa_e, ifa_h-ifa_w2, 0.0]) + tl
    stop_main = start_main + np.array([usable_x, ifa_w2, 0.0])
    # Don't append yet; we may add tip/meanders first depending on branch
    length_diff = ifa_l - usable_x

    max_length_mifa = ifa_h - mifa_meander_edge_distance-ifa_w2
    max_edgelength_tip = ifa_h - mifa_tipdistance-ifa_w2

    # --- Tip element branch ---
    # Tip starts at right end of main, goes upward (positive y) then down
    tip_anchor = stop_main + np.array([0.0, 0.0, 0.0])

    if length_diff <= max_edgelength_tip:
        # Only a partial tip is needed
        tip_start = tip_anchor
        tip_stop = tip_start + np.array([-ifa_w2, -length_diff-ifa_w2, 0.0])
        add_box_from_start_stop(tip_start, tip_stop, "tip_partial")

        add_base()
        return plates
    #print(f"length_diff={length_diff*1e3:.2f} mm")
    # Full tip first
    tip_start = tip_anchor
    tip_stop = tip_start + np.array([-ifa_w2, -max_edgelength_tip-ifa_w2, 0.0])
    add_box_from_start_stop(tip_start, tip_stop, "tip_full")
    
    length_diff -= max_edgelength_tip
    #print(f"max_edgelength_tip={max_edgelength_tip*1e3:.2f} mm used, remaining length_diff={length_diff*1e3:.2f} mm")
    # --- Meanders for remaining length ---
    # We meander down/up in y by fractions of max_length_mifa, stepping in x by (mifa_meander+ifa_w2)
    #print(f"max_length_mifa={max_length_mifa*1e3:.2f} mm")
    if length_diff > 0:
        ldiff_ratio = length_diff / (max_length_mifa * 2.0)  # each meander adds this much normalized length
        # Continue from tip bottom-left edge
        current_stop = tip_start + np.array([0.0, 0.0, 0.0])

        while ldiff_ratio > 0:
            current_meander = min(1.0, ldiff_ratio)
            ldiff_ratio -= current_meander
            #print(f"  adding meander with ratio {current_meander:.3f}, remaining ldiff_ratio={ldiff_ratio:.3f}")
            if current_meander < 0.05*mm:
                break

            # Top horizontal of meander (leftwards)
            seg1_start = current_stop + np.array([0.0, 0.0, 0.0])
            seg1_stop  = seg1_start + np.array([-mifa_meander-ifa_w2, -ifa_w2, 0.0])
            add_box_from_start_stop(seg1_start, seg1_stop, "meander_top")

            # Down leg
            seg2_start = seg1_stop + np.array([0.0, ifa_w2, 0.0])
            seg2_stop  = seg2_start + np.array([ifa_w2, -current_meander * max_length_mifa-ifa_w2, 0.0])
            add_box_from_start_stop(seg2_start, seg2_stop, "meander_down")

            # Bottom horizontal (rightwards)
            seg3_start = seg2_stop + np.array([0.0, ifa_w2, 0.0])
            seg3_stop  = seg3_start + np.array([-mifa_meander - ifa_w2, -ifa_w2, 0.0])
            add_box_from_start_stop(seg3_start, seg3_stop, "meander_bottom")

            # Up leg
            seg4_start = seg3_stop + np.array([0, 0, 0.0])
            seg4_stop  = seg4_start + np.array([+ifa_w2, current_meander * max_length_mifa+ifa_w2, 0.0])
            add_box_from_start_stop(seg4_start, seg4_stop, "meander_up")

            current_stop = seg4_stop +  np.array([0.0, 0.0, 0.0])# tail for the next loop
            
        add_box_from_start_stop(start_main, current_stop + np.array([0, 0, 0.0]), "base_link")

    # Add the straight base last (if not already linked fully)
    #add_base()
    return plates

def add_feedstub(parameters, fp_origin):
    """Add feed stub to the model at given feed point origin."""
    ifa_stub = parameters["ifa_fp"]-parameters["ifa_e"]
    ifa_short_circuit_stub= em.geo.XYPlate(
        parameters["ifa_w1"], 
        parameters["ifa_h"] + 2*parameters["via_size"],   
        position=fp_origin + np.array([-ifa_stub, 
                                       -2*parameters["via_size"], 0.0]))
    return ifa_short_circuit_stub

def add_ss_via(parameters, fp_origin,via_name="via"):
    ifa_stub = parameters["ifa_fp"]-parameters["ifa_e"]
    via_coord = em.CoordinateSystem(xax = (1,0,0),yax = (0,1,0),zax = (0,0,1),origin=fp_origin + np.array([-ifa_stub+parameters["ifa_w1"]/2, -parameters["via_size"], 0]))
    via = em.geo.Cylinder(parameters['via_size']/2, -parameters['board_th'], cs=via_coord,name=via_name)
    return via

def build_mifa(p,
               model=None,
               view_mesh=False,
               view_model=False,
               run_simulation=False,
               compute_farfield=False,
               loglevel="ERROR",
               solver=em.EMSolver.PARDISO,
               validate_ifa_antenna=True,
               return_skeleton=False):
    if model is None:
        model = em.Simulation('PatchAntenna', loglevel=loglevel)
        model.set_solver(solver)
        model.check_version("1.1.0") # Checks version compatibility.

    # --- Unit and simulation parameters --------------------------------------
    mm = 0.001              # meters per millimeter

    # --- Antenna geometry dimensions ----------------------------------------
    if validate_ifa_antenna == True:
        errs, warns, drv = validate_ifa_params(p)
        if errs:
            for err in errs:
                print(f"Parameter validation error: {err}")
            raise ValueError("IFA parameter validation failed.")

    ifa_h = p['ifa_h']
    ifa_l = p['ifa_l'] 
    ifa_w1 = p['ifa_w1'] 
    ifa_w2 = p['ifa_w2'] 
    ifa_wf = p['ifa_wf'] 
    ifa_fp= p['ifa_fp'] 
    ifa_e = p['ifa_e'] #right side edge clearance
    ifa_e2 = p['ifa_e2'] #left side edge clearance
    ifa_te = p['ifa_te'] #top edge clearance

    ifa_stub = p['ifa_fp']-p['ifa_e']

    via_size = p['via_size'] 
    mifa_meander = p['mifa_meander'] 
    
    mifa_meander_edge_distance = p['mifa_meander_edge_distance'] 
    mifa_tipdistance = p.get('mifa_tipdistance', mifa_meander_edge_distance)

    board_wsub = p['board_wsub']          # substrate width
    board_hsub = p['board_hsub']          # substrate length
    board_th = p['board_th']          # substrate thickness
    
    # Refined frequency range for antenna resonance around 1.54–1.6 GHz
    f1 = p['f1']             # start frequency
    f2 = p['f2']             # stop frequency
    freq_points = p['freq_points']           # number of frequency points
    
    mesh_boundary_size_divisor = p['mesh_boundary_size_divisor']
    mesh_wavelength_fraction = p['mesh_wavelength_fraction']
    
    
    # --- Define geometry primitives -----------------------------------------
    # Substrate block centered at origin in XY, thickness in Z (negative down)
    dielectric = em.geo.Box(p['board_wsub'], p['board_hsub'], p['board_th'],
                            position=(-p['board_wsub']/2, -p['board_hsub']/2, -p['board_th']))

    lambda1 = em.lib.C0 / ((f1))*p.get('lambda_scale',1)
    lambda2 = em.lib.C0 / ((f2))*p.get('lambda_scale',1)
    # Asymmetric margins (scale if you need to shrink/grow the domain)
    fwd     = 0.50*lambda2   #in antenna direction
    back    = 0.30*lambda2   #behind PCB
    sideL   = 0.30*lambda2   #each side
    sideR   = sideL
    top     = 0.30*lambda2   #above MIFA tip
    bot     = 0.30*lambda2   #below PCB

    Rair    = 0.5*lambda2+board_hsub/2   # air sphere radius

    # Air box dimensions & placement (assume PCB spans x∈[0, pcbL], y∈[-pcbW/2, +pcbW/2], z≈0..mifaH)
    airX = p['board_hsub'] + fwd + back
    airY = p['board_wsub'] + sideL + sideR
    airZ = top + bot + p['board_th']
    x0, y0, z0 = -sideL - p['board_wsub']/2, -back - p['board_hsub']/2, -bot - p['board_th']/2


    # Air volume around substrate (Z positive)
    #air = em.geo.Sphere(Rair).background()
    air = em.geo.Box(airY,airX, airZ, position=(x0, y0, z0)).background()

    fp_origin = np.array([-p['board_wsub']/2 + p['ifa_fp'], p['board_hsub']/2 - p['ifa_h'] - p['ifa_te'], 0.0])
        
    plates = build_mifa_plates(
        p, tl=fp_origin+np.array([0, 0, 0]), name_prefix="ifa"
    )

    ifa_feed_stub         = em.geo.XYPlate(p['ifa_wf'], p['ifa_h'] + 2*p['via_size'],       position=fp_origin + np.array([0.0, -2*p['via_size'], 0.0]))
    ifa_short_circuit_stub = add_feedstub(p, fp_origin)
    # ifa_radiating_element = em.geo.XYPlate(ifa_l,  ifa_w2,                 position=fp_origin + np.array([-ifa_stub,  ifa_h - ifa_w2, 0.0]))

    via = add_ss_via(p, fp_origin)

    ground = em.geo.XYPlate(p['board_wsub'], fp_origin[1]+p['board_hsub']/2, position=(-p['board_wsub']/2, -p['board_hsub']/2, -p['board_th'])).set_material(em.lib.PEC)


    # Plate defining lumped port geometry (origin + width/height vectors)
    port = em.geo.Plate(
        fp_origin+np.array([0, -2*p['via_size'], 0]),  # lower port corner
        np.array([p['ifa_wf'], 0, 0]),                # width vector along X
        np.array([0, 0, -p['board_th']])                    # height vector along Z
    )
    
    ifa = plates[0]
    for plate in plates[1:]:
        ifa = em.geo.add(ifa, plate)

    # # Build final ifa shape
    ifa = em.geo.add(ifa, ifa_feed_stub)
    ifa = em.geo.add(ifa, ifa_short_circuit_stub)

    if return_skeleton:
        return model, ifa, via, dielectric, air, port, ground
    
    ifa.set_material(em.lib.PEC)
    via.set_material(em.lib.PEC)
    # --- Assign materials and simulation settings ---------------------------
    # Dielectric material with some transparency for display
    dielectric.material = em.Material(3.38, color="#207020", opacity=0.9)

    model.commit_geometry()

    model.mw.set_resolution(p['mesh_wavelength_fraction'])
    model.mw.set_frequency_range(p['f1'], p['f2'], p['freq_points'])

    smallest_instance = min(p['ifa_w2'], p['ifa_wf'], p['ifa_w1'])
    smallest_via = min(p['via_size'], p['board_th'])
    smallest_port = min(p['ifa_wf'], p['board_th'])

    model.mesher.set_boundary_size(ifa, smallest_instance*p['mesh_boundary_size_divisor'])
    model.mesher.set_boundary_size(via, smallest_via*p['mesh_boundary_size_divisor'])
    model.mesher.set_face_size(port, smallest_port*p['mesh_boundary_size_divisor'])

    # --- Generate mesh and preview ------------------------------------------
    model.mesher.set_algorithm(em.Algorithm3D.HXT)
    model.generate_mesh()
    
    if view_mesh:
        model.view(selections=[port], plot_mesh=True,volume_mesh=False)              # show the mesh around the port
    if view_model:
        model.view()
        
    if not run_simulation:
        return model, None, None,None,None,None
        
    # --- Boundary conditions ------------------------------------------------
    # Define lumped port with specified orientation and impedance
    port_bc = model.mw.bc.LumpedPort(
    port, 1,
    width=p['ifa_wf'], height=p['board_th'],
    direction=em.ZAX, Z0=50
    )

    # Predefining selection
    # The outside of the air box for the absorbing boundary
    boundary_selection = air.boundary()
    # The patch and ground surface for PEC
    pec_selection = em.select(ifa,ground)

    # Assigning the boundary conditions
    abc = model.mw.bc.AbsorbingBoundary(boundary_selection)
    # --- Run frequency-domain solver ----------------------------------------
    data = model.mw.run_sweep()

    # --- Post-process S-parameters ------------------------------------------
    freqs = data.scalar.grid.freq
    freq_dense = np.linspace(p['f1'], p['f2'], 1001)
    
    S11 = data.scalar.grid.model_S(1, 1, freq_dense)
    
    if compute_farfield is False:
        return model, S11,freq_dense, None,None,None
    
    # --- Far-field radiation pattern ----------------------------------------
    # Extract 2D cut at phi=0 plane and plot E-field magnitude
    ff1 = data.field.find(freq=2.45e9)\
        .farfield_2d((0, 0, 1), (1, 0, 0), boundary_selection)
    ff2 = data.field.find(freq=2.45e9)\
        .farfield_2d((0, 0, 1), (0, 1, 0), boundary_selection)
    model.display.add_object(ifa)
    model.display.add_object(via)
    model.display.add_object(dielectric)
    # Compute full 3D far-field and display surface colored by |E|
    ff3d = data.field.find(freq=2.45e9).farfield_3d(boundary_selection)

    return model, S11,freq_dense, ff1,ff2,ff3d


def build_dualfreq_mifa(parameters = None,
    model=None,
    view_mesh=False,
    view_model=False,
    run_simulation=True,
    compute_farfield=True,
    loglevel="INFO",solver=em.EMSolver.CUDSS):

    if parameters is None or not isinstance(parameters, dict):
        raise ValueError("Invalid parameters")

    p = parameters.get("p", {})
    p2 = parameters.get("p2", {})

    if model is None:
        model = em.Simulation('PatchAntenna', loglevel=loglevel)
        model.set_solver(solver)
        model.check_version("1.1.0") # Checks version compatibility.

    model, ifa, via,dielectric,air,port,ground = build_mifa(p,model,return_skeleton=True)

    fp_origin = np.array([-p['board_wsub']/2 + p['ifa_fp'], p['board_hsub']/2 - p['ifa_h'] - p['ifa_te'], 0.0])

    fp_origin2 = fp_origin#+np.array([0,0,-p['board_th']])

    ifa2_plates = build_mifa_plates(
        p2,
        tl=fp_origin2,
        name_prefix="ifa_bottom"
    )

    ifa2 = ifa2_plates[0]
    for plate in ifa2_plates[1:]:
        ifa2 = em.geo.add(ifa2, plate)

    feedstub2 = add_feedstub(p2, fp_origin2)
    ifa2 = em.geo.add(ifa2, feedstub2)
    ifa = em.geo.add(ifa, ifa2)

    via2=add_ss_via(p2, fp_origin2, via_name="via_p2")   

    ifa.set_material(em.lib.PEC)
    via.set_material(em.lib.PEC)
    via2.set_material(em.lib.PEC)
    # --- Assign materials and simulation settings ---------------------------
    # Dielectric material with some transparency for display
    dielectric.material = em.Material(3.38, color="#207020", opacity=0.9)

    model.commit_geometry()

    model.mw.set_resolution(p['mesh_wavelength_fraction'])
    model.mw.set_frequency_range(p['f1'], p['f2'], p['freq_points'])

    smallest_instance = min(p['ifa_w2'], p['ifa_wf'], p['ifa_w1'],p2['ifa_w2'], p2['ifa_w1'])
    smallest_via = min(p['via_size'], p['board_th'])
    smallest_via2 = min(p2['via_size'], p2['board_th'])
    smallest_port = min(p['ifa_wf'], p['board_th'])
    
    model.mesher.set_boundary_size(ifa, smallest_instance*p['mesh_boundary_size_divisor'])
    model.mesher.set_boundary_size(via, smallest_via*p['mesh_boundary_size_divisor'])
    model.mesher.set_boundary_size(via2, smallest_via2*p['mesh_boundary_size_divisor'])
    model.mesher.set_face_size(port, smallest_port*p['mesh_boundary_size_divisor'])

    # --- Generate mesh and preview ------------------------------------------
    model.mesher.set_algorithm(em.Algorithm3D.HXT)
    model.generate_mesh()

    if view_mesh:
        model.view(selections=[port], plot_mesh=True,volume_mesh=False)              # show the mesh around the port
    if view_model:
        model.view()
        
    if not run_simulation:
        return model, None, None,None,None,None
        
    # --- Boundary conditions ------------------------------------------------
    # Define lumped port with specified orientation and impedance
    port_bc = model.mw.bc.LumpedPort(
    port, 1,
    width=p['ifa_wf'], height=p['board_th'],
    direction=em.ZAX, Z0=50
    )

    # Predefining selection
    # The outside of the air box for the absorbing boundary
    boundary_selection = air.boundary()
    # The patch and ground surface for PEC
    pec_selection = em.select(ifa,ground)

    # Assigning the boundary conditions
    abc = model.mw.bc.AbsorbingBoundary(boundary_selection)
    # --- Run frequency-domain solver ----------------------------------------
    data = model.mw.run_sweep()

    # --- Post-process S-parameters ------------------------------------------
    freqs = data.scalar.grid.freq
    freq_dense = np.linspace(p['f1'], p['f2'], 1001)

    S11 = data.scalar.grid.model_S(1, 1, freq_dense)

    if compute_farfield is False:
        return model, S11,freq_dense, None,None,None

    # --- Far-field radiation pattern ----------------------------------------
    # Extract 2D cut at phi=0 plane and plot E-field magnitude
    ff1 = data.field.find(freq=p['f0'])\
        .farfield_2d((0, 0, 1), (1, 0, 0), boundary_selection)
    ff2 = data.field.find(freq=p['f0'])\
        .farfield_2d((0, 0, 1), (0, 1, 0), boundary_selection)
    model.display.add_object(ifa)
    model.display.add_object(via)
    model.display.add_object(via2)
    model.display.add_object(dielectric)
    # Compute full 3D far-field and display surface colored by |E|
    ff3d = data.field.find(freq=p['f0']).farfield_3d(boundary_selection)

    return model, S11,freq_dense, ff1,ff2,ff3d

def get_loss_at_freq(S11,f0,freq_dense):
    """Compute return loss (dB) from S11 complex values."""
    # If you need to interpolate complex S11 first, do real & imag separately:
    S11_re = np.interp(f0, freq_dense, S11.real)
    S11_im = np.interp(f0, freq_dense, S11.imag)
    S11_f0 = S11_re + 1j*S11_im

    # Return loss (positive dB number)
    RL_dB = -20*np.log10(np.abs(S11_f0))
    return RL_dB

def get_resonant_frequency(S11, freq_dense):
    """Get resonant frequency (minimum |S11|) in the S11 data."""
    RL_dB = -20*np.log10(np.abs(S11))
    idx_min = np.argmax(np.abs(RL_dB))
    f_resonant = freq_dense[idx_min]
    return f_resonant

def get_bandwidth(S11, freq_dense, rl_threshold_dB=10.0, f0=None):
    """Get bandwidth (Hz) at given return loss threshold (dB) arouind f0"""
    RL_dB = -20*np.log10(np.abs(S11))
    if f0 is None:
        f0 = get_resonant_frequency(S11, freq_dense)
    # Find indices where RL crosses threshold
    indices_below = np.where(RL_dB <= -rl_threshold_dB)[0]
    if len(indices_below) == 0:
        return [0.0, 0.0]  # No bandwidth found

    # Find closest points below threshold on either side of f0
    freqs_below = freq_dense[indices_below]
    left_indices = indices_below[freqs_below < f0]
    right_indices = indices_below[freqs_below > f0]

    if len(left_indices) == 0 or len(right_indices) == 0:
        return [0.0, 0.0]  # No valid bandwidth found

    f_left = freq_dense[left_indices[-1]]
    f_right = freq_dense[right_indices[0]]

    bandwidth = np.array([f_left,f_right])
    return bandwidth


def get_s11_at_freq(S11,f0,freq_dense):
    """Get S11 complex value at center frequency."""
    # If you need to interpolate complex S11 first, do real & imag separately:
    S11_re = np.interp(f0, freq_dense, S11.real)
    S11_im = np.interp(f0, freq_dense, S11.imag)
    S11_f0 = S11_re + 1j*S11_im
    return S11_f0