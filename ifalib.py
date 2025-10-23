from pyexpat import model
import emerge as em
import numpy as np

def build_mifa_plates(
    *,
    # --- board / antenna params (all in meters) ---
    substrate_width,          # total PCB width (x extent available for the radiator)
    substrate_thickness,      # PCB thickness (not used here, but often handy)
    ifa_l,                    # total desired electrical length (projected along x)
    ifa_h,                    # height of radiator above ground (y)
    ifa_w2,                   # radiator strip width
    ifa_fp,                   # feed offset from board origin along x (see your sketch)
    ifa_e,                    # edge clearance along x (keep-out from left)
    ifa_e2,                   # edge clearance along x (keep-out from right)
    ifa_te,                   # edge clearance along y (keep-out from top)
    # --- meander/tip geometry knobs ---
    mifa_meander,             # horizontal meander step (x)
    mifa_meander_edge_distance,  # y-clearance from ground edge for meanders
    mifa_tipdistance,         # y-clearance for the tip element
    # --- placement ---
    tl=np.array([0.0, 0.0, 0.0]),  # global translation vector
    # --- bookkeeping ---
    name_prefix="ifa",
    priority=10
):
    """
    Returns a list of em.geo.XYPlate objects representing the IFA radiator geometry
    (no meshing calls). Coordinates follow your ASCII sketch; z=0 plane.
    """
    mm = 0.001  # meters per millimeter
    plates = []

    def add_box_from_start_stop(start, stop, tag):
        """Convert 'start/stop' corners into an XYPlate and append it."""
        x0, y0, _ = start
        x1, y1, _ = stop
        pos = (min(x0, x1), min(y0, y1), 0.0)
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

    if ifa_l < usable_x:
        # Simple straight radiator
        stop_main = start_main + np.array([ifa_l, -ifa_w2, 0.0])
        add_box_from_start_stop(start_main, stop_main, "main")
        return plates  # done

    # We need tip (vertical) + potential meanders
    start_main = np.array([-ifa_fp+ifa_e, ifa_h, 0.0]) + tl
    stop_main = start_main + np.array([usable_x, -ifa_w2, 0.0])
    # Don't append yet; we may add tip/meanders first depending on branch
    length_diff = ifa_l - usable_x

    max_length_mifa = ifa_h - mifa_meander_edge_distance -ifa_w2
    max_edgelength_tip = ifa_h - mifa_tipdistance

    # --- Tip element branch ---
    # Tip starts at right end of main, goes upward (positive y) then down
    tip_anchor = stop_main + np.array([0.0, +ifa_w2, 0.0])

    if length_diff < max_edgelength_tip:
        # Only a partial tip is needed
        tip_start = tip_anchor
        tip_stop = tip_start + np.array([-ifa_w2, -length_diff - ifa_w2, 0.0])
        add_box_from_start_stop(tip_start, tip_stop, "tip_partial")

        add_base()
        return plates

    # Full tip first
    tip_start = tip_anchor
    tip_stop = tip_start + np.array([-ifa_w2, -max_edgelength_tip, 0.0])
    add_box_from_start_stop(tip_start, tip_stop, "tip_full")
    length_diff -= max_edgelength_tip

    # --- Meanders for remaining length ---
    # We meander down/up in y by fractions of max_length_mifa, stepping in x by (mifa_meander+ifa_w2)
    if length_diff > 0:
        ldiff_ratio = length_diff / (max_length_mifa * 2.0 + mifa_meander)  # each meander adds this much normalized length
        # Continue from tip bottom-left edge
        current_stop = tip_start + np.array([-ifa_w2, 0.0, 0.0])

        while ldiff_ratio > 0:
            current_meander = min(1.0, ldiff_ratio)
            ldiff_ratio -= current_meander
            if current_meander < 0.05*mm:
                break

            # Top horizontal of meander (leftwards)
            seg1_start = current_stop + np.array([ifa_w2, 0.0, 0.0])
            seg1_stop  = seg1_start + np.array([-mifa_meander - ifa_w2, -ifa_w2, 0.0])
            add_box_from_start_stop(seg1_start, seg1_stop, "meander_top")

            # Down leg
            seg2_start = seg1_stop + np.array([0.0, ifa_w2, 0.0])
            seg2_stop  = seg2_start + np.array([ifa_w2, -current_meander * max_length_mifa-ifa_w2, 0.0])
            add_box_from_start_stop(seg2_start, seg2_stop, "meander_down")

            # Bottom horizontal (rightwards)
            seg3_start = seg2_stop
            seg3_stop  = seg3_start + np.array([-mifa_meander - ifa_w2, ifa_w2, 0.0])
            add_box_from_start_stop(seg3_start, seg3_stop, "meander_bottom")

            # Up leg
            seg4_start = seg3_stop + np.array([+ifa_w2, -ifa_w2, 0.0])
            seg4_stop  = seg4_start + np.array([-ifa_w2, current_meander * max_length_mifa+ifa_w2, 0.0])
            add_box_from_start_stop(seg4_start, seg4_stop, "meander_up")

            current_stop = seg4_stop  # tail for the next loop
        add_box_from_start_stop(start_main, current_stop + np.array([ifa_w2, -ifa_w2, 0.0]), "base_link")

    # Add the straight base last (if not already linked fully)
    #add_base()
    return plates

def build_mifa(params,
               model=None,
               view_mesh=False,
               view_model=False,
               run_simulation=True,
               compute_farfield=True,
               loglevel="ERROR",
               solver=em.EMSolver.PARDISO):

    if model is None:
        model = em.Simulation('PatchAntenna', loglevel=loglevel)
        model.set_solver(solver)
        model.check_version("1.1.0") # Checks version compatibility.

    # --- Unit and simulation parameters --------------------------------------
    mm = 0.001              # meters per millimeter

    # --- Antenna geometry dimensions ----------------------------------------

    ifa_h = params['ifa_h'] 
    ifa_l = params['ifa_l'] 
    ifa_w1 = params['ifa_w1'] 
    ifa_w2 = params['ifa_w2'] 
    ifa_wf = params['ifa_wf'] 
    ifa_fp= params['ifa_fp'] 
    ifa_e = params['ifa_e'] #right side edge clearance
    ifa_e2 = params['ifa_e2'] #left side edge clearance
    ifa_te = params['ifa_te'] #top edge clearance
    
    ifa_stub = ifa_fp-ifa_e
    
    via_size = params['via_size'] 
    mifa_meander = params['mifa_meander'] 
    mifa_meander_edge_distance = params['mifa_meander_edge_distance'] 
    mifa_tipdistance = params['mifa_tipdistance'] 

    wsub = params['wsub']          # substrate width
    hsub = params['hsub']          # substrate length
    th = params['th']          # substrate thickness

    # Refined frequency range for antenna resonance around 1.54–1.6 GHz
    f1 = params['f1']             # start frequency
    f2 = params['f2']             # stop frequency
    freq_points = params['freq_points']           # number of frequency points
    
    boundry_size_divisor = params['boundry_size_divisor']
    wavelength_fraction = params['wavelength_fraction']
    
    # --- Define geometry primitives -----------------------------------------
    # Substrate block centered at origin in XY, thickness in Z (negative down)
    dielectric = em.geo.Box(wsub, hsub, th,
                            position=(-wsub/2, -hsub/2, -th))

    lambda1 = em.lib.C0 / ((f1))*params['lambda_scale']
    lambda2 = em.lib.C0 / ((f2))*params['lambda_scale']
    # Asymmetric margins (scale if you need to shrink/grow the domain)
    fwd     = 0.50*lambda2   #in antenna direction
    back    = 0.30*lambda2   #behind PCB
    sideL   = 0.30*lambda2   #each side
    sideR   = sideL
    top     = 0.30*lambda2   #above MIFA tip
    bot     = 0.30*lambda2   #below PCB

    Rair    = 0.5*lambda2+hsub/2   # air sphere radius

    # Air box dimensions & placement (assume PCB spans x∈[0, pcbL], y∈[-pcbW/2, +pcbW/2], z≈0..mifaH)
    airX = hsub + fwd + back
    airY = wsub + sideL + sideR
    airZ = top + bot+th 
    x0, y0, z0 =  -sideL-wsub/2, -back-hsub/2, -bot-th/2


    # Air volume around substrate (Z positive)
    #air = em.geo.Sphere(Rair).background()
    air = em.geo.Box(airY,airX, airZ, position=(x0, y0, z0)).background()

    fp_origin = np.array([-wsub/2 + ifa_fp, hsub/2 - ifa_h - ifa_te, 0.0])
        
    plates = build_mifa_plates(
        substrate_width=wsub, substrate_thickness=th,
        ifa_l=ifa_l, ifa_h=ifa_h, ifa_w2=ifa_w2, ifa_fp=ifa_fp, ifa_e=ifa_e,ifa_e2=ifa_e2,
        ifa_te=ifa_te, mifa_meander=mifa_meander, mifa_meander_edge_distance=mifa_meander_edge_distance,
        mifa_tipdistance=mifa_tipdistance, tl=fp_origin+np.array([0, 0, 0]), name_prefix="ifa"
    )

    ifa_feed_stub         = em.geo.XYPlate(ifa_wf, ifa_h + 2*via_size,       position=fp_origin + np.array([0.0, -2*via_size, 0.0]))
    ifa_short_circuit_stub= em.geo.XYPlate(ifa_w1, ifa_h + 2*via_size,   position=fp_origin + np.array([-ifa_stub, -2*via_size, 0.0]))
    # ifa_radiating_element = em.geo.XYPlate(ifa_l,  ifa_w2,                 position=fp_origin + np.array([-ifa_stub,  ifa_h - ifa_w2, 0.0]))

    via_coord = em.CoordinateSystem(xax = (1,0,0),yax = (0,1,0),zax = (0,0,1),origin=fp_origin + np.array([-ifa_stub+ifa_w1/2, -via_size, 0]))
    via = em.geo.Cylinder(via_size/2, -th, cs=via_coord)

    ground = em.geo.XYPlate(wsub, fp_origin[1]+hsub/2, position=(-wsub/2, -hsub/2, -th)).set_material(em.lib.PEC)


    # Plate defining lumped port geometry (origin + width/height vectors)
    port = em.geo.Plate(
        fp_origin+np.array([0, -2*via_size, 0]),  # lower port corner
        np.array([ifa_wf, 0, 0]),                # width vector along X
        np.array([0, 0, -th])                    # height vector along Z
    )
    ifa = plates[0]
    for p in plates[1:]:
        ifa = em.geo.add(ifa, p)

    # # Build final ifa shape
    ifa = em.geo.add(ifa, ifa_feed_stub)
    ifa = em.geo.add(ifa, ifa_short_circuit_stub)
    ifa.set_material(em.lib.PEC)
    via.set_material(em.lib.PEC)
    # --- Assign materials and simulation settings ---------------------------
    # Dielectric material with some transparency for display
    dielectric.material = em.Material(3.38, color="#207020", opacity=0.9)
    
    model.commit_geometry()
    
    model.mw.set_resolution(wavelength_fraction)
    model.mw.set_frequency_range(f1, f2, freq_points)
        
    smallest_instance = min(ifa_w2, ifa_wf, ifa_w1)
    smallest_via = min(via_size, th)
    smallest_port = min(ifa_wf, th)
    
    model.mesher.set_boundary_size(ifa, smallest_instance*boundry_size_divisor)
    model.mesher.set_boundary_size(via, smallest_via*boundry_size_divisor)
    model.mesher.set_face_size(port, smallest_port*boundry_size_divisor)

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
    width=ifa_wf, height=th,
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
    freq_dense = np.linspace(f1, f2, 1001)
    
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

def get_loss(S11,f0,freq_dense):
    """Compute return loss (dB) from S11 complex values."""
    # If you need to interpolate complex S11 first, do real & imag separately:
    S11_re = np.interp(f0, freq_dense, S11.real)
    S11_im = np.interp(f0, freq_dense, S11.imag)
    S11_f0 = S11_re + 1j*S11_im

    # Return loss (positive dB number)
    RL_dB = -20*np.log10(np.abs(S11_f0))
    return RL_dB

def get_s11_at_freq(S11,f0,freq_dense):
    """Get S11 complex value at center frequency."""
    # If you need to interpolate complex S11 first, do real & imag separately:
    S11_re = np.interp(f0, freq_dense, S11.real)
    S11_im = np.interp(f0, freq_dense, S11.imag)
    S11_f0 = S11_re + 1j*S11_im
    return S11_f0