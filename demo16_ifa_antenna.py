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
#|           |   |    ___  _____________________| w3 | |
#|         ifa_h |   |   ||                          | |                                 
#|           |   |   |   ||                          | |                       
#|           |   |   |   ||                          | |                    
#|___________|___|___|___||__________________________| |    
#| <---ifa_e---->| w1|   wf\                         | |
#|               |_stub__|  \                        | |
#|                       |    feed point             | |
#|                       |                           | |                   
#|<-------ifa_fp-------->|                           | |
#|                                                   | |
#|___________________________________________________| |
# \___________________________________________________\|
############################################################################


"""

# --- Unit and simulation parameters --------------------------------------
mm = 0.001              # meters per millimeter

# --- Antenna geometry dimensions ----------------------------------------

ifa_h = 6.5 * mm
ifa_l = 18.7 * mm
ifa_w1 = 0.9 * mm
ifa_w2 = 0.8 * mm
ifa_wf = 0.8 * mm
ifa_fp= 2.5 * mm
ifa_stub = 2 * mm
ifa_e = ifa_fp-ifa_stub
ifa_te = 0.5 * mm
via_size = 0.5 * mm

board_wsub = 22 * mm         # substrate width
board_hsub = 40 * mm         # substrate length
board_th = 1.5 * mm         # substrate thickness

# Refined frequency range for antenna resonance around 1.54–1.6 GHz
f1 = 2.3e9             # start frequency
f2 = 2.6e9             # stop frequency
freq_points = 5           # number of frequency points

# --- Create simulation object -------------------------------------------
model = em.Simulation('PatchAntenna', loglevel='DEBUG')
model.set_solver(em.EMSolver.CUDSS)
model.check_version("1.1.0") # Checks version compatibility.

# --- Define geometry primitives -----------------------------------------
# Substrate block centered at origin in XY, thickness in Z (negative down)
dielectric = em.geo.Box(board_wsub, board_hsub, board_th,
                        position=(-board_wsub/2, -board_hsub/2, -board_th))

lambda1 = em.lib.C0 / ((f1))
lambda23 = em.lib.C0 / ((f2))
# Asymmetric margins (scale if you need to shrink/grow the domain)
fwd     = 0.50*lambda1   #in antenna direction
back    = 0.30*lambda1   #behind PCB
sideL   = 0.30*lambda1   #each side
sideR   = sideL
top     = 0.30*lambda1   #above MIFA tip
bot     = 0.30*lambda1   #below PCB

Rair    = 0.5*lambda1+board_hsub/2   # air sphere radius

# Air box dimensions & placement (assume PCB spans x∈[0, pcbL], y∈[-pcbW/2, +pcbW/2], z≈0..mifaH)
airX = board_hsub + fwd + back
airY = board_wsub + sideL + sideR
airZ = top + bot+board_th 
x0, y0, z0 =  -sideL-board_wsub/2, -back-board_hsub/2, -bot-board_th/2


# Air volume around substrate (Z positive)
#air = em.geo.Sphere(Rair).background()
air = em.geo.Box(airY,airX, airZ, position=(x0, y0, z0)).background()

fp_origin = np.array([-board_wsub/2 + ifa_fp, board_hsub/2 - ifa_h - ifa_te, 0.0])

ifa_feed_stub         = em.geo.XYPlate(ifa_wf, ifa_h + via_size,       position=fp_origin + np.array([0.0, -1.5*via_size, 0.0]))
ifa_short_circuit_stub= em.geo.XYPlate(ifa_w2, ifa_h + 1.5*via_size,   position=fp_origin + np.array([-ifa_stub, -1.5*via_size, 0.0]))
ifa_radiating_element = em.geo.XYPlate(ifa_l,  ifa_w2,                 position=fp_origin + np.array([-ifa_stub,  ifa_h - ifa_w2, 0.0]))

via_coord = em.CoordinateSystem(xax = (1,0,0),yax = (0,1,0),zax = (0,0,1),origin=fp_origin + np.array([-ifa_stub+ifa_w2/2, -via_size, 0]))
via = em.geo.Cylinder(via_size/2, -board_th, cs=via_coord)

ground = em.geo.XYPlate(board_wsub, fp_origin[1]+board_hsub/2, position=(-board_wsub/2, -board_hsub/2, -board_th)).set_material(em.lib.PEC)


# Plate defining lumped port geometry (origin + width/height vectors)
port = em.geo.Plate(
    fp_origin+np.array([0, -1.5*via_size, 0]),  # lower port corner
    np.array([ifa_wf, 0, 0]),                # width vector along X
    np.array([0, 0, -board_th])                    # height vector along Z
)

# Build final ifa shape
ifa = em.geo.add(ifa_feed_stub, ifa_radiating_element)
ifa = em.geo.add(ifa, ifa_short_circuit_stub)
ifa.set_material(em.lib.PEC)
via.set_material(em.lib.PEC)
# --- Assign materials and simulation settings ---------------------------
# Dielectric material with some transparency for display
dielectric.material = em.Material(3.38, color="#207020", opacity=0.9)

# Mesh resolution: fraction of wavelength
model.mw.set_resolution(0.2)

# Frequency sweep across the resonance
model.mw.set_frequency_range(f1, f2, freq_points)

# --- Combine geometry into simulation -----------------------------------
model.commit_geometry()

# --- Mesh refinement settings --------------------------------------------
# Finer boundary mesh on patch edges for accuracy
model.mesher.set_boundary_size(ifa, 0.2 * mm)
model.mesher.set_boundary_size(via, 0.2 * mm)
# Refined mesh on port face for excitation accuracy
model.mesher.set_face_size(port, 0.2 * mm)

# --- Generate mesh and preview ------------------------------------------
model.mesher.set_algorithm(em.Algorithm3D.HXT)
model.generate_mesh()   
# build the finite-element mesh
#model.view()
#model.view(selections=[port], plot_mesh=True)              # show the mesh around the port

# --- Boundary conditions ------------------------------------------------
# Define lumped port with specified orientation and impedance
port_bc = model.mw.bc.LumpedPort(
    port, 1,
    width=ifa_wf, height=board_th,
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
