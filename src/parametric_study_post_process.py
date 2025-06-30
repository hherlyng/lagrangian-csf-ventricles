import ufl

import numpy   as np
import dolfinx as dfx
import colormaps as cm
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py    import MPI
from basix.ufl import element
from matplotlib import lines

plt.style.use('fast')

# Facet tags
AQUEDUCT_TOP = 45
AQUEDUCT_BOT = 58

# Conversion factors
pa_to_mmhg = 1/133.3 # Pascal [Pa] to millimeters Mercury [mmHg]
m3_to_ml = 1e6 # Meters cubed [m^3] to milliliters [ml]

T = 3.0
dt = 0.001
N = int(T / dt)
period = 1
times = dt*np.arange(0, int(period / dt))

comm = MPI.COMM_WORLD
mesh_prefix = 'medium'
solver_type = 'navier-stokes'
model_variations = ['deformation']
moduli = [500, 1500, 3000]
degrees = [1]

infile_0 = f"../output/ex3/{mesh_prefix}-mesh/flow-E={moduli[0]}-k={degrees[0]}/{solver_type}/checkpoints/BDM_{model_variations[0]}_velocity_T={T}_dt={dt:.4g}"

# Initial setup of mesh
mesh = a4d.read_mesh(filename=infile_0, comm=comm)
ft   = a4d.read_meshtags(filename=infile_0, mesh=mesh, meshtag_name='ft')
n = ufl.FacetNormal(mesh)
dS = ufl.Measure('dS', domain=mesh, subdomain_data=ft)

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

# Prepare figures
fig, ax = plt.subplots(figsize=[16, 9])
ax_ = ax.twinx()
fig2, ax2 = plt.subplots(figsize=[16, 9])
linestyles = ['-', '--', '-.', ':']
plot_idx = 0
flowrate_colors = cm.bubblegum.discrete(3).colors
deltap_colors   = cm.emerald.discrete(4).colors[1:]
labels = []
for j, E in enumerate(moduli):
    for k in degrees:
        # Initialize finite elements and functions
        bdm_el = element("BDM", mesh.basix_cell(), k)
        dg_vec_el = element("DG", mesh.basix_cell(), k, shape=(mesh.geometry.dim,))
        dg_el  = element("DG", mesh.basix_cell(), k-1)
        V = dfx.fem.functionspace(mesh, bdm_el)
        Q = dfx.fem.functionspace(mesh, dg_el)
        uh = dfx.fem.Function(V)
        ph = dfx.fem.Function(Q)

        # Initialize integral forms
        u_flux = ufl.dot(ufl.avg(uh), n('-'))
        flowrate_top_form = dfx.fem.form(u_flux*dS(AQUEDUCT_TOP))
        flowrate_bot_form = dfx.fem.form(u_flux*dS(AQUEDUCT_BOT))
        area_top_aq = dfx.fem.assemble_scalar(dfx.fem.form(1*dS(AQUEDUCT_TOP)))
        area_bot_aq = dfx.fem.assemble_scalar(dfx.fem.form(1*dS(AQUEDUCT_BOT)))
        mean_pressure_top_form = dfx.fem.form(ufl.avg(ph)*dS(AQUEDUCT_TOP))
        mean_pressure_bot_form = dfx.fem.form(ufl.avg(ph)*dS(AQUEDUCT_BOT))

        for model_variation in model_variations:
            infile_name = f"../output/ex3/{mesh_prefix}-mesh/flow-E={E}-k={k}/{solver_type}/checkpoints/BDM_{model_variation}_velocity_T={T}_dt={dt:.4g}"
            print(f"\nPost processing results from file:\n{infile_name}\n")

            flowrates_top_aq = []
            flowrates_bot_aq = []
            pressure_gradients_aq = []

            for i, t in enumerate(times):
                print(f'Time t = {t:.4g}')

                a4d.read_function(filename=infile_name, u=uh, time=i, name='relative_velocity')
                a4d.read_function(filename=infile_name, u=ph, time=i, name='pressure')

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

            # Convert lists to numpy arrays
            flowrates_top_aq = np.array(flowrates_top_aq)
            flowrates_bot_aq = np.array(flowrates_bot_aq)
            pressure_gradients_aq = np.array(pressure_gradients_aq)

            print(f'Sum of flow rates = {np.sum(flowrates_top_aq[1:]+flowrates_top_aq[:-1])/2*dt:.4g}')

            # Plot
            print("Plotting ...")
            label = f'E={E}, k={k}'
            labels.append(label)
            ax.plot(times, flowrates_top_aq, color=flowrate_colors[plot_idx], linestyle=linestyles[0], label=label)
            ax_.plot(times, pressure_gradients_aq, color=deltap_colors[plot_idx], linestyle=linestyles[1], label=label)
            ax2.plot(times, flowrates_top_aq, color=flowrate_colors[plot_idx], linestyle=linestyles[0], label=label)
            plot_idx += 1

# Configure plots
ax.set_ylabel('ml/s', fontsize=40)
ax.tick_params(axis='both', labelsize=30)
ax.set_xlabel('Time [s]', fontsize=40) 
ax.legend(labels, fontsize=20, loc='upper right', frameon=True, fancybox=False, edgecolor='k')
ax_.legend(labels, fontsize=20, loc='upper right', frameon=True, fancybox=False, edgecolor='k')
ax_.set_ylabel('mmHg/m', color=deltap_colors[-1], fontsize=40)
ax_.tick_params(axis='y', colors=deltap_colors[-1], labelsize=30)
ax2.set_ylabel('ml/s', fontsize=40)
ax2.tick_params(axis='both', labelsize=30)
ax2.legend(labels, fontsize=20, loc='upper right', frameon=True, fancybox=False, edgecolor='k')

fig.suptitle(rf"Aqueduct $\Delta P$ and flowrate, mesh={mesh_prefix}, solver={solver_type}")
fig2.suptitle(f"Median aperture flowrate, mesh={mesh_prefix}, solver={solver_type}")
fig.tight_layout()
fig2.tight_layout()

save_figs = 1
if save_figs:
    fig.savefig(f"../output/illustrations/flowrates_pressure-gradients_aqueduct_mesh={mesh_prefix}_solver={solver_type}")
    fig2.savefig(f"../output/illustrations/flowrates_median_aperture_mesh={mesh_prefix}_solver={solver_type}")

plt.show()