import ufl

import numpy   as np
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from scifem    import assemble_scalar
from mpi4py    import MPI
from basix.ufl import element

plt.style.use('fast')

# Facet tags
AQUEDUCT_TOP = 45
AQUEDUCT_BOT = 58

# Conversion factors
pa_to_mmhg = 1/133.3 # Pascal [Pa] to millimeters Mercury [mmHg]
m3_to_ml = 1e6 # Meters cubed [m^3] to milliliters [ml]

k = 1 # Element degree

comm = MPI.COMM_WORLD
mesh_prefix = 'medium'
solver_type = 'navier-stokes'
infile_name = f'../output/{mesh_prefix}-mesh/flow/{solver_type}/checkpoints/BDM_deforming_velocity/'
mesh = a4d.read_mesh(filename=infile_name, comm=comm, read_from_partition=True)
# ft   = a4d.read_meshtags(filename=infile_name, mesh=mesh, meshtag_name='ft')
mesh.topology.create_entities(mesh.topology.dim-1)
with dfx.io.XDMFFile(comm, '../geometries/medium_ventricles_mesh_tagged.xdmf', 'r') as xdmf:
    ft = xdmf.read_meshtags(mesh, "ft")

bdm_el = element("BDM", mesh.basix_cell(), k)
dg_el  = element("DG", mesh.basix_cell(), k-1)
V = dfx.fem.functionspace(mesh, bdm_el)
Q = dfx.fem.functionspace(mesh, dg_el)
uh = dfx.fem.Function(V)
ph = dfx.fem.Function(Q)
n = ufl.FacetNormal(mesh)
u_flux = ufl.dot(uh, n)

# Compute cross-sectional areas and length of aqueduct
dS = ufl.Measure('dS', domain=mesh, subdomain_data=ft)
area_top_aq = assemble_scalar(1*dS(AQUEDUCT_TOP))
area_bot_aq = assemble_scalar(1*dS(AQUEDUCT_BOT))
facet_top_aq = ft.find(AQUEDUCT_TOP)[0]
facet_bot_aq = ft.find(AQUEDUCT_BOT)[0]
mesh.topology.create_connectivity(mesh.topology.dim-1, 0)
f_to_v = mesh.topology.connectivity(mesh.topology.dim-1, 0)
vertex_top_aq = f_to_v.links(facet_top_aq)[0]
vertex_bot_aq = f_to_v.links(facet_bot_aq)[0]
point_top_aq = mesh.geometry.x[vertex_top_aq, :]
point_bot_aq = mesh.geometry.x[vertex_bot_aq, :]
length_aq = np.sqrt(np.sum((point_top_aq-point_bot_aq)**2))

T = 2
dt = 0.01
N = int(T / dt)
times = np.linspace(0, T, N+1)
times = times[1:]

flowrates_top_aq = []
flowrates_bot_aq = []
pressure_gradients_aq = []

for t in times:
    print(f'Time t = {t:.4g}')

    a4d.read_function(filename=infile_name, u=uh, time=t, name='uh')
    # a4d.read_function(filename=infile_name, u=ph, time=t, name='ph')

    # Calculate flow rates
    flowrate_top_aq = assemble_scalar(u_flux('+')*dS(AQUEDUCT_TOP))*m3_to_ml
    flowrate_bot_aq = assemble_scalar(u_flux('+')*dS(AQUEDUCT_BOT))*m3_to_ml

    # Calculate pressure gradient in the aqueduct
    mean_pressure_top_aq = 1/area_top_aq*assemble_scalar(ph('+')*dS(AQUEDUCT_TOP))
    mean_pressure_bot_aq = 1/area_bot_aq*assemble_scalar(ph('+')*dS(AQUEDUCT_BOT))
    delta_pressure_aq = (mean_pressure_bot_aq-mean_pressure_top_aq)/length_aq*pa_to_mmhg

    # Print and append
    print(f'Flowrate top aqueduct: \t{flowrate_top_aq:.4g}')
    print(f'Flowrate bot aqueduct: \t{flowrate_bot_aq:.4g}')
    print(f'Pressure gradient aqueduct: {delta_pressure_aq:.4g}')
    [l.append(val) for l, val in zip([flowrates_top_aq, flowrates_bot_aq, pressure_gradients_aq], 
                                     [flowrate_top_aq,  flowrate_bot_aq,  delta_pressure_aq])]

# Convert lists to numpy arrays
flowrates_top_aq = np.array(flowrates_top_aq)
flowrates_bot_aq = np.array(flowrates_bot_aq)
pressure_gradients_aq = np.array(pressure_gradients_aq)

print(f'Sum of flow rates = {np.sum(flowrates_top_aq[1:]+flowrates_top_aq[:-1])/2*dt:.4g}')

fig, ax = plt.subplots(figsize=[16, 9])
pl1, = ax.plot(times, flowrates_top_aq, color='k', label='flowrate')
ax.set_ylabel('ml/s', fontsize=40)
ax.tick_params(axis='both', labelsize=30)

ax2 = ax.twinx()
pl2, = ax2.plot(times, pressure_gradients_aq, color='r', label='pressure gradient')
ax2.set_ylabel('mmHg/m', color=pl2.get_color(), fontsize=40)
ax2.tick_params(axis='y', colors=pl2.get_color(), labelsize=30)

ax.set_xlabel('Time [s]', fontsize=40) 
ax.legend([pl1, pl2], [pl1.get_label(), pl2.get_label()],
           fontsize=20, loc='upper right', frameon=True, fancybox=False, edgecolor='k')
fig.tight_layout()
plt.show()