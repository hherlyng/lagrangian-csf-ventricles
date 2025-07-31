import sys
import numpy     as np
import colormaps as cm
import matplotlib.pyplot as plt

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
times = dt*np.arange(0, int(periods / dt)+1)
k = int(sys.argv[2])
p = int(sys.argv[3])
input_dir = f"../output/mesh_{mesh_suffix}/flow_p={p}_E={E}_k={k}_dt={dt:.4g}_T={T:.0f}/{solver_type}/"
infile_0 = input_dir+f"checkpoints/BDM_{model_version}_velocity"


infile_name = input_dir+f"checkpoints/BDM_{model_version}_velocity"
print(f"\nReading post-processed results from simulation output:\n{infile_name}\n")

data = np.load(input_dir+'postprocessed_results.npz')
flowrates_aq=data['flowrates_aq']
flowrates_ap=data['flowrates_ap']
flowrates_sc=data['flowrates_sc']
flowrates_rfm=data['flowrates_rfm']
flowrates_lfm=data['flowrates_lfm']
net_flowrates=data['net_flowrates']
cumulative_aq=data['cumulative_aq']
cumulative_ap=data['cumulative_ap']
cumulative_sc=data['cumulative_sc']
cumulative_rfm=data['cumulative_rfm']
cumulative_lfm=data['cumulative_lfm']
pressures_top_aq=data['pressures_top_aq']
pressures_rfm=data['pressures_rfm']
pressures_lfm=data['pressures_lfm']
pressure_gradients_aq=data['pressure_gradients_aq']
max_velocity=data['max_velocity']
max_velocity_time=data['max_velocity_time']
min_velocity=data['min_velocity']
min_velocity_time=data['min_velocity_time']
dP=data['dP']
Re=data['Re']

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
print(f"Max Reynolds number = {Re}")

# Plot
print("Plotting ...")

# Prepare figures
linestyles = ['-', '--', '-.', ':']
flowrate_colors = cm.haline.discrete(6).colors
deltap_colors   = cm.thermal.discrete(4).colors
ls_idx = 0
label_end = ''#f'E={E}, k={k}, p={p}'
lw = 4
me = int(len(times)/25) # Marker interval
ms = 12 # Marker size
params = {
    "lines.linewidth" : lw,
    "legend.fontsize" : 30,
    "text.usetex" : True,
    "font.family" : "serif",
    "axes.labelsize" :  50,
    "xtick.labelsize" : 40,
    "ytick.labelsize" : 40
}
plt.rcParams.update(params)

figsize = [20, 11]
fig, ax = plt.subplots(figsize=figsize)
ax_ = ax.twinx()
fig2, ax2 = plt.subplots(figsize=figsize)
fig3, ax3 = plt.subplots(figsize=figsize)
fig4, ax4 = plt.subplots(figsize=figsize)

# Plot flowrate and pressure gradient in the aqueduct
ax.plot(times, flowrates_aq, color=flowrate_colors[0], linestyle=linestyles[ls_idx], linewidth=lw, label="Flowrate ")
ax_.plot(times, pressure_gradients_aq, color=deltap_colors[2], linestyle=linestyles[ls_idx+1],
         linewidth=lw,  marker='^', markersize=ms, markevery=me, label=r"$\Delta P$ "+label_end)

# Plot flowrates
ax2.plot(times, flowrates_aq, color=flowrate_colors[0], linestyle=linestyles[ls_idx],
         linewidth=lw,  label="Aqueduct "+label_end)
ax2.plot(times, flowrates_ap, color=flowrate_colors[1], linestyle=linestyles[ls_idx],
         linewidth=lw,  marker='x', markersize=ms, markevery=me, label="Apertures")
ax2.plot(times, flowrates_sc, color=flowrate_colors[2], linestyle=linestyles[ls_idx],
         linewidth=lw,  marker='o', markersize=ms, markevery=me, label="Canal")
ax2.plot(times, flowrates_rfm, color=flowrate_colors[3], linestyle=linestyles[ls_idx],
         linewidth=lw,  marker='^', markersize=ms, markevery=me, label=r"Right LV$\rightarrow$3V foramen")
ax2.plot(times, flowrates_lfm, color=flowrate_colors[4], linestyle=linestyles[ls_idx],
         linewidth=lw,  marker='s', markersize=ms, markevery=me, label=r"Left LV$\rightarrow$3V foramen")

# Plot cumulative flow volumes
ax3.plot(times, cumulative_aq*1000, color=flowrate_colors[0], linestyle=linestyles[ls_idx],
         linewidth=lw,  label="Aqueduct "+label_end)
ax3.plot(times, cumulative_ap*1000, color=flowrate_colors[1], linestyle=linestyles[ls_idx],
         linewidth=lw,  marker='x', markersize=ms, markevery=me, label="Apertures")
ax3.plot(times, cumulative_sc*1000, color=flowrate_colors[2], linestyle=linestyles[ls_idx],
         linewidth=lw,  marker='o', markersize=ms, markevery=me, label="Canal")
ax3.plot(times, cumulative_rfm*1000, color=flowrate_colors[3], linestyle=linestyles[ls_idx],
         linewidth=lw,  marker='^', markersize=ms, markevery=me, label=r"Right LV$\rightarrow$3V foramen ")
ax3.plot(times, cumulative_lfm*1000, color=flowrate_colors[4], linestyle=linestyles[ls_idx],
         linewidth=lw,  marker='s', markersize=ms, markevery=me, label=r"Left LV$\rightarrow$3V foramen ")

# Plot pressures
ax4.plot(times, pressures_top_aq, color=deltap_colors[0], linestyle=linestyles[ls_idx+1],
         linewidth=lw,  label=r"$\overline{p}$ aqueduct "+label_end)
ax4.plot(times, pressures_rfm, color=deltap_colors[1], linestyle=linestyles[ls_idx+1], 
         linewidth=lw,  marker='^', markersize=ms, markevery=me, label=r"$\overline{p}$ right LV$\rightarrow$3V foramen ")
ax4.plot(times, pressures_lfm, color=deltap_colors[2], linestyle=linestyles[ls_idx+1],
         linewidth=lw,  marker='s', markersize=ms, markevery=me, label=r"$\overline{p}$ left LV$\rightarrow$3V foramen ")

# Configure plots
ax.set_ylabel('ml/s')
ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='k')
ax_.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='k')
ax_.set_ylabel('mmHg/cm', color=deltap_colors[2])
ax_.tick_params(axis='y', colors=deltap_colors[2])
ax2.set_ylabel('ml/s')
ax2.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='k')
ax3.set_ylabel('microliters')
ax3.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='k')
ax4.set_ylabel('mmHg')
ax4.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='k')
[axes.set_xlabel('Time [s]') for axes in [ax, ax2, ax3, ax4]]

[figure.tight_layout() for figure in [fig, fig2, fig3, fig4]]

save_figs = 1
fig_dir = "../output/illustrations/flow/"
if save_figs:
    fig.savefig(fig_dir+f"aqueduct_flowrate_pressure_gradients_p={p}_k={k}_E={E}_T={T}_mesh={mesh_suffix}_solver={solver_type}_model={model_version}")
    fig2.savefig(fig_dir+f"flowrates_p={p}_k={k}_E={E}_T={T}_mesh={mesh_suffix}_solver={solver_type}_model={model_version}")
    fig3.savefig(fig_dir+f"flow_volumes_p={p}_k={k}_E={E}_T={T}_mesh={mesh_suffix}_solver={solver_type}_model={model_version}")
    fig4.savefig(fig_dir+f"pressures_p={p}_k={k}_E={E}_T={T}_mesh={mesh_suffix}_solver={solver_type}_model={model_version}")

    plot_V_change = True
    if plot_V_change:
        fig5, ax5 = plt.subplots(figsize=figsize)
        dV_dt = np.load(f"../output/mesh_{mesh_suffix}/deformation_p={p}_E={E}_k={k}_T={T}/"+"dV_dt.npy")
        ax5.plot(times[1:], net_flowrates[1:], color='k', linewidth=lw,  label='net flow')
        ax5.plot(times[1:], -dV_dt[:int(len(times)-1)]*1e-3, color='r', linewidth=lw,  label='-dV/dt')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('ml/s')
        ax5.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='k')
        fig5.tight_layout()
        fig5.savefig(fig_dir+f"net_flow_p={p}_k={k}_E={E}_T={T}_mesh={mesh_suffix}_solver={solver_type}_model={model_version}")
        
print("Post processing complete.")
plt.show()