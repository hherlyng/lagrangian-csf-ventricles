import adios4dolfinx
import mpi4py
import dolfinx
import ufl
import sys
import numpy as np
import matplotlib.pyplot as plt
import basix.ufl

mesh_suffix = int(sys.argv[1])
T = int(sys.argv[2])
p = int(sys.argv[3])
k = int(sys.argv[4])
E = 1500
dt = 0.001
output_dir = "/path/to/output_dir" \
            + f"/output/mesh_{mesh_suffix}/deformation_p={p}_E={E}_k={k}_T={T}/"
cpoint_filename = output_dir+f"checkpoints/displacement_velocity_dt={dt:.4g}"
mesh = adios4dolfinx.read_mesh(cpoint_filename, mpi4py.MPI.COMM_WORLD)
ct = adios4dolfinx.read_meshtags(cpoint_filename, mesh, "ct")
cg_el = basix.ufl.element("CG", mesh.basix_cell(), p, shape=(3,))

V = dolfinx.fem.functionspace(mesh, cg_el)
v = dolfinx.fem.Function(V)

r = ufl.SpatialCoordinate(mesh)
chi = r + v
F = ufl.grad(chi)
J = ufl.det(F)

dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)

times = np.arange(1001)

m3_to_mul = 1e9 # Meters cubed [m^3] to microliters [mu l]

# Mesh tags
LATERAL_VENTRICLES = 7
THIRD_VENTRICLE    = 4
FOURTH_VENTRICLE   = 8

vol_tot_form = dolfinx.fem.form(1*J*dx)
vol_LV_form = dolfinx.fem.form(1*J*dx(LATERAL_VENTRICLES))
vol_3V_form = dolfinx.fem.form(1*J*dx(THIRD_VENTRICLE))
vol_4V_form = dolfinx.fem.form(1*J*dx(FOURTH_VENTRICLE))

volumes_tot = []
volumes_LV = []
volumes_3V = []
volumes_4V = []

for t in times:
    adios4dolfinx.read_function(filename=cpoint_filename, u=v, time=t, name="defo_displacement")
    vol_tot = dolfinx.fem.assemble_scalar(vol_tot_form)*m3_to_mul
    vol_LV = dolfinx.fem.assemble_scalar(vol_LV_form)*m3_to_mul
    vol_3V = dolfinx.fem.assemble_scalar(vol_3V_form)*m3_to_mul
    vol_4V = dolfinx.fem.assemble_scalar(vol_4V_form)*m3_to_mul
    print("Volume total: ", vol_tot)
    print("Volume LV: ", vol_LV)
    print("Volume 3V: ", vol_3V)
    print("Volume 4V: ", vol_4V)
    if t==0:
        initial_vol_tot = vol_tot
        initial_vol_LV = vol_LV
        initial_vol_3V = vol_3V
        initial_vol_4V = vol_4V
    else:
        print("Percentage change total volume: ", (vol_tot/initial_vol_tot-1)*100)
        print("Percentage change LV volume: ", (vol_LV/initial_vol_LV-1)*100)
        print("Percentage change 3V volume: ", (vol_3V/initial_vol_3V-1)*100)
        print("Percentage change 4V volume: ", (vol_4V/initial_vol_4V-1)*100)
    volumes_tot.append(vol_tot)
    volumes_LV.append(vol_LV)
    volumes_3V.append(vol_3V)
    volumes_4V.append(vol_4V)

volumes_tot = np.array(volumes_tot)/initial_vol_tot
volumes_LV  = np.array(volumes_LV)/initial_vol_LV
volumes_3V  = np.array(volumes_3V)/initial_vol_3V
volumes_4V  = np.array(volumes_4V)/initial_vol_4V
dV_dt = np.array(volumes_tot[1:] - volumes_tot[:-1])/dt*initial_vol_tot
times_plot = times*dt

fig, ax = plt.subplots(figsize=[16, 9])
ax.plot(times_plot, volumes_tot, color='k', label="Total")
ax.plot(times_plot, volumes_LV, color='r', label="LV")
ax.plot(times_plot, volumes_3V, color='b', label="3V")
ax.plot(times_plot, volumes_4V, color='g', label="4V")
ax.set_ylabel(r'Volume [$\mu$l]', fontsize=35)
ax.tick_params(axis='both', labelsize=25)
ax.set_xlabel('Time [s]', fontsize=35) 
ax.legend(fontsize=20, loc='upper left', frameon=True, fancybox=False, edgecolor='k')
fig_dir = "/global/D1/homes/hherlyng/lagrangian-csf-ventricles/output/illustrations/"
fig.suptitle(f"Relative volume change, mesh={mesh_suffix}")
fig.tight_layout()
fig.savefig(fig_dir+f"volumes_p={p}_E={E}_T={T}_mesh={mesh_suffix}")

fig2, ax2 = plt.subplots(figsize=[16, 9])
ax2.plot(times_plot[1:], dV_dt, color='k', label="dV/dt")
ax2.set_ylabel(r'Volume change per time [$\mu$l/s]', fontsize=35)
ax2.tick_params(axis='both', labelsize=25)
ax2.set_xlabel('Time [s]', fontsize=35) 
ax2.legend(fontsize=20, loc='upper left', frameon=True, fancybox=False, edgecolor='k')
fig2.suptitle(f"Time rate of change volume, mesh={mesh_suffix}")
fig2.tight_layout()
fig2.savefig(fig_dir+f"volume_change_p={p}_E={E}_T={T}_mesh={mesh_suffix}")

# Save data
np.savez_compressed(output_dir+"/data/volumes.npz",
        volumes_tot = volumes_tot,
        volumes_LV  = volumes_LV,
        volumes_3V  = volumes_3V,
        volumes_4V  = volumes_4V
)
np.save(output_dir+"/data/dV_dt.npy", dV_dt)