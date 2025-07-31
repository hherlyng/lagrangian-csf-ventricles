import ufl
import sys
import numpy   as np
import dolfinx as dfx
import adios4dolfinx     as a4d

from mpi4py    import MPI
from basix.ufl import element

# Facet tags
CANAL_OUT = 23
LATERAL_APERTURES = 28
AQUEDUCT_TOP = 45
AQUEDUCT_BOT = 58
LATERAL_VENTRICLES_FORAMINA = 67

# Conversion factors
pa_to_mmhg = 1/133.3 # Pascal [Pa] to millimeters Mercury [mmHg]
m3_to_ml = 1e6 # Meters cubed [m^3] to milliliters [ml]

comm = MPI.COMM_WORLD
models = {1 : "deformation",
          2 : "deformation+cilia",
          3 : "deformation+production",
          4 : "deformation+cilia+production"
}
mesh_suffix = int(sys.argv[6])
solver_type = str(sys.argv[5])
model_version = models[int(sys.argv[4])]
E = 1500
T = int(sys.argv[1])
dt = 0.001
N = int(T / dt)
periods = 1
times = dt*np.arange(0, int(periods / dt)+1)[:3]
k = int(sys.argv[2])
p = int(sys.argv[3])
input_dir = f"../output/mesh_{mesh_suffix}/flow_p={p}_E={E}_k={k}_dt={dt:.4g}_T={T:.0f}/{solver_type}/"
infile_0 = input_dir+f"checkpoints/BDM_{model_version}_velocity"

# Initial setup of mesh
mesh = a4d.read_mesh(filename=infile_0, comm=comm)
ft   = a4d.read_meshtags(filename=infile_0, mesh=mesh, meshtag_name='ft')
n = ufl.FacetNormal(mesh)
ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft) # External facet integral
dS = ufl.Measure('dS', domain=mesh, subdomain_data=ft) # Internal facet integral

# Compute cross-sectional areas and length of aqueduct
area_bot_aq = dfx.fem.assemble_scalar(dfx.fem.form(1*dS(AQUEDUCT_BOT)))
area_top_aq = dfx.fem.assemble_scalar(dfx.fem.form(1*dS(AQUEDUCT_TOP)))
xx, yy, zz = ufl.SpatialCoordinate(mesh)

x_greater = ufl.conditional(ufl.gt(xx, 0), 1.0, 0.0)
x_lower = ufl.conditional(ufl.lt(xx, 0), 1.0, 0.0)
area_rfm = dfx.fem.assemble_scalar(dfx.fem.form(x_greater*dS(LATERAL_VENTRICLES_FORAMINA))) # Right foramen area
area_lfm = dfx.fem.assemble_scalar(dfx.fem.form(x_lower*dS(LATERAL_VENTRICLES_FORAMINA))) # Left foramen area
facet_bot_aq = ft.find(AQUEDUCT_BOT)[0]
facet_top_aq = ft.find(AQUEDUCT_TOP)[0]
mesh.topology.create_connectivity(mesh.topology.dim-1, 0)
f_to_v = mesh.topology.connectivity(mesh.topology.dim-1, 0)
vertex_top_aq = f_to_v.links(facet_top_aq)[0]
vertex_bot_aq = f_to_v.links(facet_bot_aq)[0]
point_top_aq = mesh.geometry.x[vertex_top_aq, :]
point_bot_aq = mesh.geometry.x[vertex_bot_aq, :]
length_aq = np.sqrt(np.sum((point_top_aq-point_bot_aq)**2))
diam = np.sqrt(4*area_top_aq/np.pi)
print("Length of aqueduct [m]: ", length_aq)
print("Diameter aqueduct top (circular equivalent) [m]: ", diam)

# Initialize finite elements and functions
bdm_el = element("BDM", mesh.basix_cell(), k)
dg_vec_el = element("DG", mesh.basix_cell(), k, shape=(mesh.geometry.dim,))
dg_el  = element("DG", mesh.basix_cell(), k-1)
V = dfx.fem.functionspace(mesh, bdm_el)
U = dfx.fem.functionspace(mesh, dg_vec_el)
Q = dfx.fem.functionspace(mesh, dg_el)
uh = dfx.fem.Function(V)
uh_dg = dfx.fem.Function(U)
ph = dfx.fem.Function(Q)

# Initialize integral forms
u_flux = ufl.dot(uh('-'), n('-'))
u_flux_bdry = ufl.dot(uh, n)
flowrate_aq_form = dfx.fem.form(u_flux*dS(AQUEDUCT_BOT))
flowrate_ap_form = dfx.fem.form(u_flux_bdry*ds(LATERAL_APERTURES))
flowrate_sc_form = dfx.fem.form(u_flux_bdry*ds(CANAL_OUT))
flowrate_rfm_form = dfx.fem.form(x_greater*u_flux*dS(LATERAL_VENTRICLES_FORAMINA))
flowrate_lfm_form = dfx.fem.form(x_lower*u_flux*dS(LATERAL_VENTRICLES_FORAMINA))
mean_pressure_top_form = dfx.fem.form(ufl.avg(ph)*dS(AQUEDUCT_TOP))
mean_pressure_bot_form = dfx.fem.form(ufl.avg(ph)*dS(AQUEDUCT_BOT))
mean_pressure_rfm_form  = dfx.fem.form(x_greater*ufl.avg(ph)*dS(LATERAL_VENTRICLES_FORAMINA))
mean_pressure_lfm_form  = dfx.fem.form(x_lower*ufl.avg(ph)*dS(LATERAL_VENTRICLES_FORAMINA))

infile_name = input_dir+f"checkpoints/BDM_{model_version}_velocity"
print(f"\nPost processing results from file:\n{infile_name}\n")

max_velocity = -np.inf
min_velocity = -np.inf

flowrates_aq = []
flowrates_ap = []
flowrates_sc = []
flowrates_rfm = []
flowrates_lfm = []
net_flowrates = []
pressures_top_aq = []
pressures_rfm = []
pressures_lfm = []
pressure_gradients_aq = []

for i, t in enumerate(times):
    print(f'Time t = {t:.4g}')

    a4d.read_function(filename=infile_name, u=uh, time=i, name='relative_velocity')
    a4d.read_function(filename=infile_name, u=ph, time=i, name='pressure')

    uh_dg.interpolate(uh)
    uh_reshaped = uh_dg.x.array.copy().reshape(-1, 3)
    negative_mask = uh_reshaped[:, 2] > 0 # Velocities that have a positive z component are considered negative flow
    min_magnitude = np.linalg.norm(uh_reshaped[negative_mask], axis=1).max()

    positive_mask = uh_reshaped[:, 2] < 0 # Velocities that have a negative z component are considered positive flow
    max_magnitude = np.linalg.norm(uh_reshaped[positive_mask], axis=1).max()
    if max_magnitude.size > 0:
        if max_magnitude > max_velocity:
            max_velocity = max_magnitude
            max_velocity_time = t
    if min_magnitude.size > 0:
        # Only check for minimum in the case that there are velocity
        # vectors with a negative z component
        if min_magnitude > min_velocity:
            min_velocity = min_magnitude
            min_velocity_time = t
    
    # Calculate flow rates
    flowrate_aq = comm.allreduce(dfx.fem.assemble_scalar(flowrate_aq_form)*m3_to_ml, op=MPI.SUM)
    flowrate_ap = comm.allreduce(dfx.fem.assemble_scalar(flowrate_ap_form)*m3_to_ml, op=MPI.SUM)
    flowrate_sc = comm.allreduce(dfx.fem.assemble_scalar(flowrate_sc_form)*m3_to_ml, op=MPI.SUM)
    flowrate_lfm = comm.allreduce(dfx.fem.assemble_scalar(flowrate_lfm_form)*m3_to_ml, op=MPI.SUM)
    flowrate_rfm = comm.allreduce(dfx.fem.assemble_scalar(flowrate_rfm_form)*m3_to_ml, op=MPI.SUM)
    net_flow = flowrate_ap + flowrate_sc

    # Calculate pressure gradient in the aqueduct
    mean_pressure_top_aq = comm.allreduce(1/area_top_aq*dfx.fem.assemble_scalar(mean_pressure_top_form), op=MPI.SUM)*pa_to_mmhg
    mean_pressure_bot_aq = comm.allreduce(1/area_bot_aq*dfx.fem.assemble_scalar(mean_pressure_bot_form), op=MPI.SUM)*pa_to_mmhg
    mean_pressure_rfm = comm.allreduce(1/area_rfm*dfx.fem.assemble_scalar(mean_pressure_rfm_form), op=MPI.SUM)*pa_to_mmhg
    mean_pressure_lfm = comm.allreduce(1/area_lfm*dfx.fem.assemble_scalar(mean_pressure_lfm_form), op=MPI.SUM)*pa_to_mmhg
    delta_pressure_aq = -(mean_pressure_bot_aq-mean_pressure_top_aq)/length_aq*1e-2 # Minus sign because dz=length_aq is negative, 1e-2 to convert to mmHg/cm

    # Print and append
    print(f'Flowrate aqueduct: \t{flowrate_aq:.2e}')
    print(f'Flowrate apertures: \t{flowrate_ap:.2e}')
    print(f'Flowrate canal: \t{flowrate_sc:.2e}')
    print(f'Pressure gradient aqueduct: {delta_pressure_aq:.2e}')
    [l.append(val) for l, val in zip([flowrates_aq, flowrates_ap, flowrates_sc, flowrates_rfm, flowrates_lfm,
                                      net_flowrates, pressures_top_aq, pressures_rfm, pressures_lfm, pressure_gradients_aq], 
                                    [flowrate_aq,  flowrate_ap, flowrate_sc, flowrate_rfm, flowrate_lfm,
                                    net_flow, mean_pressure_top_aq, mean_pressure_rfm, mean_pressure_lfm, delta_pressure_aq])]

# Convert lists to numpy arrays
flowrates_aq = np.array(flowrates_aq)
flowrates_ap = np.array(flowrates_ap)
flowrates_sc = np.array(flowrates_sc)
flowrates_rfm = np.array(flowrates_rfm)
flowrates_lfm = np.array(flowrates_lfm)
net_flowrates = np.array(net_flowrates)

pressures_top_aq = np.array(pressures_top_aq)
pressures_rfm = np.array(pressures_rfm)
pressures_lfm = np.array(pressures_lfm)
pressure_gradients_aq = np.array(pressure_gradients_aq)

# Calculate cumulative flow volumes
cumulative_aq = np.cumsum(flowrates_aq)*dt
cumulative_ap = np.cumsum(flowrates_ap)*dt
cumulative_sc = np.cumsum(flowrates_sc)*dt
cumulative_rfm = np.cumsum(flowrates_rfm)*dt
cumulative_lfm = np.cumsum(flowrates_lfm)*dt

# Calculate peak-to-trough pressure gradient
max_dP = np.max(pressure_gradients_aq)
min_dP = np.min(pressure_gradients_aq)
dP = max_dP - min_dP

# Calculate max Reynolds number
u_max = max(abs(max_velocity), abs(min_velocity))
rho = 1000 # Fluid density [kg/m^3]
mu  = 7e-4 # Dynamic viscosity [Pa * s]
Re = u_max * diam * rho / mu # Reynolds number

# Save the data
data_filename = input_dir+'postprocessed_results.npz'
np.savez_compressed(
    data_filename,
    flowrates_aq=flowrates_aq,
    flowrates_ap=flowrates_ap,
    flowrates_sc=flowrates_sc,
    flowrates_rfm=flowrates_rfm,
    flowrates_lfm=flowrates_lfm,
    net_flowrates=net_flowrates,
    cumulative_aq=cumulative_aq,
    cumulative_ap=cumulative_ap,
    cumulative_sc=cumulative_sc,
    cumulative_rfm=cumulative_rfm,
    cumulative_lfm=cumulative_lfm,
    pressures_top_aq=pressures_top_aq,
    pressures_rfm=pressures_rfm,
    pressures_lfm=pressures_lfm,
    pressure_gradients_aq=pressure_gradients_aq,
    max_velocity=max_velocity,
    max_velocity_time=max_velocity_time,
    min_velocity=min_velocity,
    min_velocity_time=min_velocity_time,
    dP=dP,
    Re=Re
)

print(f"Net flow volume aqueduct = {np.trapezoid(flowrates_aq, dx=dt)*1000:.3g} microliters")
print(f"Net flow volume boundaries = {np.trapezoid(net_flowrates, dx=dt)*1000:.3g} microliters")
print(f"Maximum flowrate aqueduct = {np.max(flowrates_aq):.3g} ml/s, at time t = {np.argmax(flowrates_aq)*dt:.4g}")
print(f"Minimum flowrate aqueduct = {np.min(flowrates_aq):.3g} ml/s, at time t = {np.argmin(flowrates_aq)*dt:.4g}")
print(f"Maximum velocity magnitude = {max_velocity*100:.3g} cm/s, at time t = {max_velocity_time:.4g}")
print(f"Minimum velocity magnitude = {-min_velocity*100:.3g} cm/s, at time t = {min_velocity_time:.4g}")
print(f"Stroke volume aqueduct = {np.max(cumulative_aq)*1000:.3g} microliters")
print(f"Stroke volume apertures = {np.max(cumulative_ap)*1000:.3g} microliters")
print(f"Stroke volume canal = {np.max(cumulative_sc)*1000:.3g} microliters")
print(f"Stroke volume right foramina of Monro = {np.max(cumulative_rfm)*1000:.3g} microliters")
print(f"Stroke volume left  foramina of Monro = {np.max(cumulative_lfm)*1000:.3g} microliters")
print(f"Max to min pressure gradient difference = {dP:.3g} mmHg/cm")
print("Post processing complete.")
print(f"Data saved to {data_filename}")