import emerge as em
import numpy as np
from emerge.plot import plot_sp, smith, plot_ff_polar, plot_ff

""" PATCH ANTENNA DEMO

This design is modeled after this Comsol Demo: https://www.comsol.com/model/microstrip-patch-antenna-11742

In this demo we build and simulate a rectangular patch antenna on a dielectric
substrate with airbox and lumped port excitation, then visualize S-parameters
and far-field radiation patterns. 

This simulation is quite heavy and might take a while to fully compute.

#############################################################
#|------------- substrate_width -------------------|
# ________________________________________________     _ substrate_thickness
#| A__ifa_te     |----------ifa_l(total length)-| |\   \-gndplane_position 
#| V              _______________     __________| | |   \_0 point
#|               |    ___  ___   |___|  ______  | | |
#|         ifa_h |   |   ||   |_________|    |  | | |_ mifa_meander_edge_distance 
#|               |   |   ||  mifa_meander    |__| | |_ mifa_tipdistance
#|               |   |   ||                   w2  | | |                  
#|_______________|___|___||_______________________| |_|
#| <---ifa_e---->| w1|   wf\                      | |
#|               |_stub__|  \                     | |
#|                       |    feed point          | |
#|                       |                        | | substrate_length
#|<-------ifa_fp-------->|                        | |
#|                                                | |
#|________________________________________________| |
# \________________________________________________\|
#############################################################


"""

# --- Unit and simulation parameters --------------------------------------
mm = 0.001              # meters per millimeter

# --- Antenna geometry dimensions ----------------------------------------

ifa_h = 7 * mm
ifa_h2 = 3 * mm
ifa_l = 20.2 * mm
ifa_l2 = 15 * mm
ifa_w1 = 0.613 * mm
ifa_w2 = 0.472 * mm
ifa_wf = 0.425 * mm
ifa_fp= 2.2 * mm
ifa_stub = 1.71 * mm
ifa_stub2 = 2.2 * mm
ifa_e = ifa_fp-ifa_stub
ifa_te = 0.5 * mm
via_size = 0.5 * mm

wsub = 21 * mm         # substrate width
hsub = 90 * mm         # substrate length
th = 0.12 * mm         # substrate thickness
Rair = 100 * mm         # air sphere radius

# Refined frequency range for antenna resonance around 1.54–1.6 GHz
f1 = 1.5e9             # start frequency
f2 = 3.5e9             # stop frequency

# --- Create simulation object -------------------------------------------
model = em.Simulation('PatchAntenna', loglevel='DEBUG')

model.check_version("1.1.0") # Checks version compatibility.

# --- Define geometry primitives -----------------------------------------
# Substrate block centered at origin in XY, thickness in Z (negative down)
dielectric = em.geo.Box(wsub, hsub, th,
                        position=(-wsub/2, -hsub/2, -th))

# Air box above substrate (Z positive)
air = em.geo.Sphere(Rair).background() 
# Background makes sure no materials of overlapping domains are overwritten

fp_origin = np.array([-wsub/2 + ifa_fp, hsub/2 - ifa_h - ifa_te, 0.0])

ifa_feed_stub         = em.geo.XYPlate(ifa_wf, ifa_h + via_size,       position=fp_origin + np.array([0.0, -1.5*via_size, 0.0]))
ifa_short_circuit_stub= em.geo.XYPlate(ifa_w2, ifa_h + 1.5*via_size,   position=fp_origin + np.array([-ifa_stub, -1.5*via_size, 0.0]))
ifa_radiating_element = em.geo.XYPlate(ifa_l,  ifa_w2,                 position=fp_origin + np.array([-ifa_stub,  ifa_h - ifa_w2, 0.0]))


via_coord = em.CoordinateSystem(xax = (1,0,0),yax = (0,1,0),zax = (0,0,1),origin=fp_origin + np.array([-ifa_stub+ifa_w2/2, -via_size, 0]))
via = em.geo.Cylinder(via_size/2, -th, cs=via_coord)

# ifa_dual_frequency = em.geo.XYPlate(ifa_l2, ifa_w2, position=(-ifa_stub2, ifa_h2, 0))
# ifa_short_circuit_stub2 = em.geo.XYPlate(ifa_w2, ifa_h2+1.5*via_size, position=(-ifa_stub2, -1.5*via_size, 0))

# via_coord2 = em.CoordinateSystem(xax = (1,0,0),yax = (0,1,0),zax = (0,0,1),origin=(-ifa_stub2+ifa_w2/2, -via_size, 0))
# via2 = em.geo.Cylinder(via_size/2, -th, cs=via_coord2)


   
ground = em.geo.XYPlate(wsub, fp_origin[1]+hsub/2, position=(-wsub/2, -hsub/2, -th)).set_material(em.lib.PEC)


# Plate defining lumped port geometry (origin + width/height vectors)
port = em.geo.Plate(
    fp_origin+np.array([0, -1.5*via_size, 0]),  # lower port corner
    np.array([ifa_wf, 0, 0]),                # width vector along X
    np.array([0, 0, -th])                    # height vector along Z
)

# Build final patch shape: subtract cutouts, add feed line
#rpatch = em.geo.remove(rpatch, cutout1)
#rpatch = em.geo.remove(rpatch, cutout2)
ifa = em.geo.add(ifa_feed_stub, ifa_radiating_element)
ifa = em.geo.add(ifa, ifa_short_circuit_stub)
# ifa = em.geo.add(ifa, ifa_dual_frequency)
# ifa = em.geo.add(ifa, ifa_short_circuit_stub2)
ifa.set_material(em.lib.PEC)
via.set_material(em.lib.PEC)
# via2.set_material(em.lib.PEC)
# --- Assign materials and simulation settings ---------------------------
# Dielectric material with some transparency for display
dielectric.material = em.Material(3.38, color="#207020", opacity=0.9)

# Mesh resolution: fraction of wavelength
model.mw.set_resolution(0.2)

# Frequency sweep across the resonance
model.mw.set_frequency_range(f1, f2, 10)

# --- Combine geometry into simulation -----------------------------------
model.commit_geometry()

# --- Mesh refinement settings --------------------------------------------
# Finer boundary mesh on patch edges for accuracy
model.mesher.set_boundary_size(ifa, 2 * mm)
model.mesher.set_boundary_size(via, 0.05 * mm)
# Refined mesh on port face for excitation accuracy
model.mesher.set_face_size(port, 0.05 * mm)

# --- Generate mesh and preview ------------------------------------------
model.mesher.set_algorithm(em.Algorithm3D.HXT)
model.generate_mesh()   
# build the finite-element mesh
model.view()
# model.view(selections=[port], plot_mesh=True)              # show the mesh around the port

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
data = model.mw.run_sweep()#multi_processing=True,n_workers=4)

# --- Post-process S-parameters ------------------------------------------
freqs = data.scalar.grid.freq
freq_dense = np.linspace(f1, f2, 1001)
S11 = data.scalar.grid.model_S(1, 1, freq_dense)            # reflection coefficient
plot_sp(freq_dense, S11)                       # plot return loss in dB
smith(S11, f=freq_dense, labels='S11')         # Smith chart of S11

# --- Far-field radiation pattern ----------------------------------------
# Extract 2D cut at phi=0 plane and plot E-field magnitude
ff1 = data.field.find(freq=2.45e9)\
    .farfield_2d((0, 0, 1), (1, 0, 0), boundary_selection)
ff2 = data.field.find(freq=2.45e9)\
    .farfield_2d((0, 0, 1), (0, 1, 0), boundary_selection)

plot_ff(ff1.ang*180/np.pi, [ff1.normE/em.lib.EISO, ff2.normE/em.lib.EISO], dB=True, ylabel='Gain [dBi]')                # linear plot vs theta
plot_ff_polar(ff1.ang, [ff1.normE/em.lib.EISO, ff2.normE/em.lib.EISO], dB=True, dBfloor=-20)          # polar plot of radiation

# --- 3D radiation visualization -----------------------------------------
# Add geometry to 3D display
model.display.add_object(ifa)
model.display.add_object(via)
#model.display.add_object(via2)
model.display.add_object(dielectric)
# Compute full 3D far-field and display surface colored by |E|
ff3d = data.field.find(freq=2.45e9).farfield_3d(boundary_selection)
surf = ff3d.surfplot('normE', rmax=60 * mm,
                      offset=(0, 0, 20 * mm))

model.display.add_surf(*surf)
model.display.show()
