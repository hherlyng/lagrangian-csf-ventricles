import numpy   as np
import matplotlib.pyplot as plt
import colormaps as cm
from pathlib import Path

file_extension = ".svg"
timestep = 0.001
p_refinement_study = False
mesh_refinement_study = True

params = {
    "lines.linewidth" : 3,
    "lines.markersize" : 12,
    "legend.fontsize" : 42,
    "axes.labelsize" :  60,
    "xtick.labelsize" : 50,
    "ytick.labelsize" : 50,
    "font.family" : "serif",
    "text.usetex" : True,
    "axes.labelpad" : 14.0,
    "legend.frameon" : False
}
plt.rcParams.update(params)

m_to_mm = 1e3 # Meter to millimeter
mesh_suffix = 0
E = 1500
T = 4
k = 1
p = 3
linestyles = ["-", "--", "-.", ":"]
markers = ["x", "o", "^", "s"]
bc_colors = cm.haline.discrete(6).colors
vol_colors = cm.thermal.discrete(4).colors
fig_dir = "../output/illustrations/deformation/"
Path(fig_dir).mkdir(parents=True, exist_ok=True)

figsize = [20, 11]
fig_bcs, ax_bcs = plt.subplots(figsize=figsize)
fig_vel, ax_vel = plt.subplots(figsize=figsize)
fig_vol, ax_vol = plt.subplots(figsize=figsize)

data_dir = f"../output/mesh_{mesh_suffix}/deformation_p={p}_E={E:.0f}_k={k}_T={T:.0f}/data/"
# Prepare data arrays
bc_cc = np.load(data_dir+f"applied_bc_corpus_callosum_dt={timestep}.npy", "r")
bc_cn = np.load(data_dir+f"applied_bc_caudate_nucleus_dt={timestep}.npy", "r")
bc_floor = np.load(data_dir+f"applied_bc_LV_3V_floor_dt={timestep}.npy", "r")
bc_3V = np.load(data_dir+f"applied_bc_3V_wall_dt={timestep}.npy", "r")
bc_horns = np.load(data_dir+f"applied_bc_inf_occ_horns_dt={timestep}.npy", "r")

vel = np.load(data_dir+f"point_velocities_dt={timestep}.npy", "r")
wh_dot = np.load(data_dir+f"point_wh_dots_dt={timestep}.npy", "r")

volumes = np.load(data_dir+f"volumes.npz")
volumes_tot = (volumes['volumes_tot']-1.0)*100
volumes_3V  = (volumes['volumes_3V']-1.0)*100
volumes_LV  = (volumes['volumes_LV']-1.0)*100

times = np.arange(bc_cc.shape[0])*timestep
me = int(len(times)/25) # Marker interval
ax_bcs.plot(times, m_to_mm*bc_cc, color=bc_colors[0], label="Corpus callosum")
ax_bcs.plot(times, m_to_mm*bc_cn, color=bc_colors[1], marker=markers[0], markevery=me, label="Caudate nucleus")
ax_bcs.plot(times, m_to_mm*bc_floor, color=bc_colors[2], marker=markers[1], markevery=me, label="LV/3V floor")
ax_bcs.plot(times, m_to_mm*bc_3V, color=bc_colors[3], marker=markers[2], markevery=me, label="3V walls")
ax_bcs.plot(times, m_to_mm*bc_horns, color=bc_colors[4], marker=markers[3], markevery=me, label="LV inf./occ. horns")

# Redefine times for one period
times = np.arange(1001)*timestep

ax_vel.plot(times, m_to_mm*vel[:len(times)], color=vol_colors[0], label="FFT velocities")
ax_vel.plot(times, m_to_mm*wh_dot[3000:], color=vol_colors[1], label=r"N-$\beta$ velocities")

ax_vol.plot(times, volumes_tot, color=vol_colors[0], label="Total volume")
ax_vol.plot(times, volumes_3V, color=vol_colors[1], label="3V volume")
ax_vol.plot(times, volumes_LV, color=vol_colors[2], label="LV volume")

[ax.legend() for ax in [ax_bcs, ax_vel, ax_vol]]

ax_bcs.set_ylabel("Displacement [mm]")
ax_vel.set_ylabel("Velocity [mm/s]")
ax_vol.set_ylabel(r"Volume change [\%]")
[ax.set_xlabel('Time [s]') for ax in [ax_bcs, ax_vel, ax_vol]]

plt.show()

[fig.tight_layout() for fig in [fig_bcs, fig_vel, fig_vol]]
fig_bcs.savefig(fig_dir+f"applied_bcs_mesh_{mesh_suffix}"+file_extension)
fig_vel.savefig(fig_dir+f"disp_velocities_mesh_{mesh_suffix}"+file_extension)
fig_vol.savefig(fig_dir+f"volumes_mesh_{mesh_suffix}"+file_extension)