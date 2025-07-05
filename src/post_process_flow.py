import ufl

import numpy   as np
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py    import MPI
from basix.ufl import element

plt.style.use('fast')

# Facet tags
AQUEDUCT_TOP = 45
AQUEDUCT_BOT = 58

# Conversion factors
pa_to_mmhg = 1/133.3 # Pascal [Pa] to millimeters Mercury [mmHg]
m3_to_ml = 1e6 # Meters cubed [m^3] to milliliters [ml]

p = 2 # CG element degree
k = 1 # BDM element degree
E = 500 # Young's modulus

T = 3.0
dt = 0.001
N = int(T / dt)
period = 1
times = dt*np.arange(0, int(period / dt))

comm = MPI.COMM_WORLD
mesh_prefix = 'fine'
solver_type = 'navier-stokes'
model_variation = 'deformation+cilia+production'
infile_name = f"../output/ex3/{mesh_prefix}-mesh/flow_p={p}_E={E}_k={k}_dt={dt:.4g}_T={T}/{solver_type}/checkpoints/BDM_{model_variation}_velocity"
mesh = a4d.read_mesh(filename=infile_name, comm=comm, read_from_partition=False)
ft   = a4d.read_meshtags(filename=infile_name, mesh=mesh, meshtag_name='ft')

bdm_el = element("BDM", mesh.basix_cell(), k)
dg_vec_el = element("DG", mesh.basix_cell(), k, shape=(mesh.geometry.dim,))
dg_el  = element("DG", mesh.basix_cell(), k-1)
V = dfx.fem.functionspace(mesh, bdm_el)
Q = dfx.fem.functionspace(mesh, dg_el)
uh = dfx.fem.Function(V)
ph = dfx.fem.Function(Q)
uh_out = dfx.fem.Function(dfx.fem.functionspace(mesh, dg_vec_el))
n = ufl.FacetNormal(mesh)
dS = ufl.Measure('dS', domain=mesh, subdomain_data=ft)

# Define integral forms
u_flux = ufl.dot(ufl.avg(uh), n('-'))
flowrate_top_form = dfx.fem.form(u_flux*dS(AQUEDUCT_TOP))
flowrate_bot_form = dfx.fem.form(u_flux*dS(AQUEDUCT_BOT))
area_top_aq = dfx.fem.assemble_scalar(dfx.fem.form(1*dS(AQUEDUCT_TOP)))
area_bot_aq = dfx.fem.assemble_scalar(dfx.fem.form(1*dS(AQUEDUCT_BOT)))
mean_pressure_top_form = dfx.fem.form(ufl.avg(ph)*dS(AQUEDUCT_TOP))
mean_pressure_bot_form = dfx.fem.form(ufl.avg(ph)*dS(AQUEDUCT_BOT))

# Compute cross-sectional areas and length of aqueduct
facet_top_aq = ft.find(AQUEDUCT_TOP)[0]
facet_bot_aq = ft.find(AQUEDUCT_BOT)[0]
mesh.topology.create_connectivity(mesh.topology.dim-1, 0)
f_to_v = mesh.topology.connectivity(mesh.topology.dim-1, 0)
vertex_top_aq = f_to_v.links(facet_top_aq)[0]
vertex_bot_aq = f_to_v.links(facet_bot_aq)[0]
point_top_aq = mesh.geometry.x[vertex_top_aq, :]
point_bot_aq = mesh.geometry.x[vertex_bot_aq, :]
length_aq = np.sqrt(np.sum((point_top_aq-point_bot_aq)**2))

flowrates_top_aq = []
flowrates_bot_aq = []
pressure_gradients_aq = []

# vtk = dfx.io.VTKFile(comm, "velocity.pvd", "w")
# vtk.write_mesh(mesh)

for i, t in enumerate(times):
    print(f'Time t = {t:.4g}')

    a4d.read_function(filename=infile_name, u=uh, time=i, name='relative_velocity')
    a4d.read_function(filename=infile_name, u=ph, time=i, name='pressure')
    # uh_out.interpolate(uh)

    # Calculate flow rates
    flowrate_top_aq = comm.allreduce(dfx.fem.assemble_scalar(flowrate_top_form)*m3_to_ml, op=MPI.SUM)
    flowrate_bot_aq = comm.allreduce(dfx.fem.assemble_scalar(flowrate_bot_form)*m3_to_ml, op=MPI.SUM)

    # Calculate pressure gradient in the aqueduct
    mean_pressure_top_aq = comm.allreduce(1/area_top_aq*dfx.fem.assemble_scalar(mean_pressure_top_form), op=MPI.SUM)
    mean_pressure_bot_aq = comm.allreduce(1/area_bot_aq*dfx.fem.assemble_scalar(mean_pressure_bot_form), op=MPI.SUM)
    delta_pressure_aq = -(mean_pressure_bot_aq-mean_pressure_top_aq)/length_aq*pa_to_mmhg # Minus sign because dz=length_aq is negative

    # Print and append
    print(f'Flowrate top aqueduct: \t{flowrate_top_aq:.4g}')
    print(f'Flowrate bot aqueduct: \t{flowrate_bot_aq:.4g}')
    print(f'Pressure gradient aqueduct: {delta_pressure_aq:.4g}')
    [l.append(val) for l, val in zip([flowrates_top_aq, flowrates_bot_aq, pressure_gradients_aq], 
                                     [flowrate_top_aq,  flowrate_bot_aq,  delta_pressure_aq])]

#     vtk.write_function(uh_out, t)
# vtk.close()
# Convert lists to numpy arrays
flowrates_top_aq = np.array(flowrates_top_aq)
flowrates_bot_aq = np.array(flowrates_bot_aq)
pressure_gradients_aq = np.array(pressure_gradients_aq)

print(f'Sum of flow rates = {np.sum(flowrates_top_aq[1:]+flowrates_top_aq[:-1])/2*dt:.4g}')

fig, ax = plt.subplots(figsize=[16, 9])
pl1, = ax.plot(times, flowrates_top_aq, color='k', label='flowrate')
ax.set_ylabel('ml/s', fontsize=40)
ax.tick_params(axis='both', labelsize=30)

ax_ = ax.twinx()
pl_, = ax_.plot(times, pressure_gradients_aq, color='r', label='pressure gradient')
ax_.set_ylabel('mmHg/m', color=pl_.get_color(), fontsize=40)
ax_.tick_params(axis='y', colors=pl_.get_color(), labelsize=30)

ax.set_xlabel('Time [s]', fontsize=40) 
ax.legend([pl1, pl_], [pl1.get_label(), pl_.get_label()],
           fontsize=20, loc='upper right', frameon=True, fancybox=False, edgecolor='k')
fig.suptitle("Flowrate aqueduct")
fig.tight_layout()


fig2, ax2 = plt.subplots(figsize=[16, 9])
pl2, = ax2.plot(times, flowrates_top_aq, color='k', label='flowrate')
ax2.set_ylabel('ml/s', fontsize=40)
ax2.tick_params(axis='both', labelsize=30)
ax2.legend([pl2], [pl2.get_label()],
           fontsize=20, loc='upper right', frameon=True, fancybox=False, edgecolor='k')
fig2.suptitle("Flowrate median aperture")
fig2.tight_layout()

save_figs = 1
if save_figs:
    fig.savefig(f"../output/illustrations/flowrate_aqueduct_mesh={mesh_prefix}_model={model_variation}_solver={solver_type}")
    fig2.savefig(f"../output/illustrations/flowrate_median_aperture_mesh={mesh_prefix}_model={model_variation}_solver={solver_type}")

plt.show()