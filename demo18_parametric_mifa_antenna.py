import emerge as em
import numpy as np
from emerge.plot import plot_sp, smith, plot_ff_polar, plot_ff

""" PATCH ANTENNA DEMO

This design is modeled after this Comsol Demo: https://www.comsol.com/model/microstrip-patch-antenna-11742

In this demo we build and simulate a rectangular patch antenna on a dielectric
substrate with airbox and lumped port excitation, then visualize S-parameters
and far-field radiation patterns. 

This simulation is quite heavy and might take a while to fully compute.

############################################################################
#|------------- substrate_width -------------------|
# ___________________________________________________  _ substrate_thickness
#| A__ifa_te     |----------ifa_l(total length)-|    |\   \-gndplane_position 
#| V         ___  ______________________________|    | |   \_0 point
#|           |   |    ___  _____________________| w2 | |
#|         ifa_h |   |   ||                          | |                                 
#|           |   |   |   ||                          | |                       
#|           |   |   |   ||                          | |                    
#|___________|___|___|___||__________________________| |    
#| <---ifa_e---->| w1|   wf\                     <-->| |
#|                       |  \                  ifa_e2| |
#|                       |    feed point             | |
#|                       |                           | |                   
#|<-------ifa_fp-------->|                           | |
#|                                                   | |
#|___________________________________________________| |
# \___________________________________________________\|
############################################################################


#############################################################
#|------------- substrate_width -------------------|
# _______________________________________________     _ substrate_thickness
#| A  ifa_e      |----------ifa_l(total length)-| |\  \-gndplane_position 
#| V____          _______________     __________| | |  \_0 point
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

def build_ifa_plates(
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
        ldiff_ratio = length_diff / (max_length_mifa * 2.0)
        # Continue from tip bottom-left edge
        current_stop = tip_start + np.array([-ifa_w2, 0.0, 0.0])

        while ldiff_ratio > 0:
            current_meander = min(1.0, ldiff_ratio)
            ldiff_ratio -= current_meander
            print(f"Adding meander segment with length {current_meander * max_length_mifa*1000:.3e} mm")
            if current_meander < 0.05*mm:
                print("Meander segment is too short, stopping.")
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

        # Finally connect back to the base at the short end
        print(f"Adding final base connection from {current_stop} to {start_main}")
        add_box_from_start_stop(start_main, current_stop + np.array([ifa_w2, -ifa_w2, 0.0]), "base_link")

    # Add the straight base last (if not already linked fully)
    #add_base()
    return plates

# --- Unit and simulation parameters --------------------------------------
mm = 0.001              # meters per millimeter

# --- Antenna geometry dimensions ----------------------------------------

ifa_h = 20 * mm
ifa_l = 160 * mm
ifa_w1 = 2 * mm
ifa_w2 = 1 * mm
ifa_wf = 1 * mm
ifa_fp= 4 * mm
ifa_e = 0.5 * mm
ifa_e2 = 0.5 * mm
ifa_te = 0.5 * mm
ifa_stub = ifa_fp-ifa_e
mifa_meander=2*mm
mifa_meander_edge_distance=2*mm
mifa_tipdistance=mifa_meander_edge_distance
via_size = 0.5 * mm

wsub = 30 * mm         # substrate width
hsub = 40 * mm         # substrate length
th = 1.5 * mm         # substrate thickness

#meshing parameters
common = 0.33
boundry_size_divisor=common #
wavelength_fraction=common  # mesh resolution as fraction of wavelength

# Refined frequency range for antenna resonance around 1.54–1.6 GHz
f1 = 0.5e9             # start frequency
f2 = 1.0e9             # stop frequency
freq_points = 5           # number of frequency points

# --- Create simulation object -------------------------------------------
model = em.Simulation('PatchAntenna', loglevel='DEBUG')
model.set_solver(em.EMSolver.CUDSS)
model.check_version("1.1.0") # Checks version compatibility.

# --- Define geometry primitives -----------------------------------------
# Substrate block centered at origin in XY, thickness in Z (negative down)
dielectric = em.geo.Box(wsub, hsub, th,
                        position=(-wsub/2, -hsub/2, -th))

lambda1 = em.lib.C0 / ((f1))
lambda2 = em.lib.C0 / ((f2))
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
    
plates = build_ifa_plates(
    substrate_width=wsub, substrate_thickness=th,
    ifa_l=ifa_l, ifa_h=ifa_h, ifa_w2=ifa_w2, ifa_fp=ifa_fp, ifa_e=ifa_e,ifa_e2=ifa_e2,
    ifa_te=ifa_te, mifa_meander=mifa_meander, mifa_meander_edge_distance=mifa_meander_edge_distance,
    mifa_tipdistance=mifa_tipdistance, tl=fp_origin+np.array([0, 0, 0]), name_prefix="ifa"
)

ifa_feed_stub         = em.geo.XYPlate(ifa_wf, ifa_h + 2*via_size,       position=fp_origin + np.array([0.0, -2*via_size, 0.0]))
ifa_short_circuit_stub= em.geo.XYPlate(ifa_w2, ifa_h + 2*via_size,   position=fp_origin + np.array([-ifa_stub, -2*via_size, 0.0]))
# ifa_radiating_element = em.geo.XYPlate(ifa_l,  ifa_w2,                 position=fp_origin + np.array([-ifa_stub,  ifa_h - ifa_w2, 0.0]))

via_coord = em.CoordinateSystem(xax = (1,0,0),yax = (0,1,0),zax = (0,0,1),origin=fp_origin + np.array([-ifa_stub+ifa_w2/2, -via_size, 0]))
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

# Mesh resolution: fraction of wavelength
model.mw.set_resolution(wavelength_fraction)

# Frequency sweep across the resonance
model.mw.set_frequency_range(f1, f2, freq_points)

# --- Combine geometry into simulation -----------------------------------
model.commit_geometry()

# --- Mesh refinement settings --------------------------------------------
# Finer boundary mesh on patch edges for accuracy
smallest_instance = min(ifa_w2, ifa_wf, ifa_w1)
smallest_via = min(via_size, th)
smallest_port = min(ifa_wf, th)
model.mesher.set_boundary_size(ifa, smallest_instance*boundry_size_divisor)
model.mesher.set_boundary_size(via, smallest_via*boundry_size_divisor)
model.mesher.set_face_size(port, smallest_port*boundry_size_divisor)

# --- Generate mesh and preview ------------------------------------------
model.mesher.set_algorithm(em.Algorithm3D.HXT)
model.generate_mesh()   
# build the finite-element mesh
model.view()
#model.view(selections=[port], plot_mesh=True,volume_mesh=False)              # show the mesh around the port

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
S11 = data.scalar.grid.model_S(1, 1, freq_dense)            # reflection coefficient
plot_sp(freq_dense, S11)                       # plot return loss in dB
smith(S11, f=freq_dense, labels='S11')         # Smith chart of S11

# # --- Far-field radiation pattern ----------------------------------------
# # Extract 2D cut at phi=0 plane and plot E-field magnitude
# ff1 = data.field.find(freq=2.45e9)\
#     .farfield_2d((0, 0, 1), (1, 0, 0), boundary_selection)
# ff2 = data.field.find(freq=2.45e9)\
#     .farfield_2d((0, 0, 1), (0, 1, 0), boundary_selection)

# plot_ff(ff1.ang*180/np.pi, [ff1.normE/em.lib.EISO, ff2.normE/em.lib.EISO], dB=True, ylabel='Gain [dBi]')                # linear plot vs theta
# plot_ff_polar(ff1.ang, [ff1.normE/em.lib.EISO, ff2.normE/em.lib.EISO], dB=True, dBfloor=-20)          # polar plot of radiation

# # --- 3D radiation visualization -----------------------------------------
# # Add geometry to 3D display
# model.display.add_object(ifa)
# model.display.add_object(via)
# model.display.add_object(dielectric)
# # Compute full 3D far-field and display surface colored by |E|
# ff3d = data.field.find(freq=2.45e9).farfield_3d(boundary_selection)
# surf = ff3d.surfplot('normE', rmax=60 * mm,
#                       offset=fp_origin,)

# model.display.add_surf(*surf)
# model.display.show()
