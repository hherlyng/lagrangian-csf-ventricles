import sys
import numpy     as np
import colormaps as cm
import matplotlib.pyplot as plt

k = int(sys.argv[1])
p = int(sys.argv[2])
solver_type = str(sys.argv[3])
mesh_suffix = int(sys.argv[4])
dt = 0.001
N = int(1 / dt)
times = dt*np.arange(0, N+1)
model_versions = ["deformation+cilia", "deformation+production", "deformation+cilia+production"]
input_dir = f"../output/mesh_{mesh_suffix}/flow_data/p={p}_k={k}_{solver_type}/"
model_abbrv = {1 : "d+c", 2 : "d+p", 3 : "d+c+p"}

# Prepare figures
linestyles = ['-', '--', '-.', ':']
markers = ['x', 'o', '^', 's']
flowrate_colors = cm.haline.discrete(6).colors
deltap_colors   = cm.thermal.discrete(4).colors
me = int(len(times)/25) # Marker interval
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

figsize = [20, 11]
fig, ax = plt.subplots(figsize=figsize)
ax_ = ax.twinx()
fig2, ax2 = plt.subplots(figsize=figsize)
fig3, ax3 = plt.subplots(figsize=figsize)
fig4, ax4 = plt.subplots(figsize=figsize)

print(f"Reading post-processed results with p={p}, k={k}, mesh={mesh_suffix}, solver={solver_type}")

for i, model in enumerate(model_versions):

    print("Model = ", model)
    # Load data
    data = np.load(input_dir+f'postprocessed_results_{model}.npz')
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
    label_end = model_abbrv[i+1]
    # Plot flowrate and pressure gradient in the aqueduct
    ax.plot(times, flowrates_aq, color=flowrate_colors[2*i], linestyle=linestyles[i], label="Flowrate ")
    ax_.plot(times, pressure_gradients_aq, color=deltap_colors[i], linestyle=linestyles[i],
            marker=markers[i], markevery=me, label=r"$\Delta P$ "+label_end)

    # Plot flowrates
    ax2.plot(times, flowrates_aq, color=flowrate_colors[2*i], linestyle=linestyles[i],
            label="Aqueduct "+label_end)
    ax2.plot(times, flowrates_ap, color=flowrate_colors[2*i], linestyle=linestyles[i],
            marker='x', markevery=me, label="Apertures" if i==0 else None)
    ax2.plot(times, flowrates_sc, color=flowrate_colors[2*i], linestyle=linestyles[i],
            marker='o', markevery=me, label="Canal" if i==0 else None)

    # Plot cumulative flow volumes
    ax3.plot(times, cumulative_aq*1000, color=flowrate_colors[2*i], linestyle=linestyles[i],
            label="Aqueduct "+label_end if i==0 else label_end)
    ax3.plot(times, cumulative_ap*1000, color=flowrate_colors[2*i], linestyle=linestyles[i],
            marker='x', markevery=me, label="Apertures" if i==0 else None)
    ax3.plot(times, cumulative_sc*1000, color=flowrate_colors[2*i], linestyle=linestyles[i],
            marker='o', markevery=me, label="Canal" if i==0 else None)
    ax3.plot(times, cumulative_rfm*1000, color=flowrate_colors[2*i], linestyle=linestyles[i],
            marker='^', markevery=me, label=r"Right LV$\rightarrow$3V foramen" if i==0 else None)

    # Plot pressures
    ax4.plot(times, pressures_top_aq, color=deltap_colors[i], linestyle=linestyles[i],
            label=r"$\overline{p}$ aqueduct "+label_end if i==0 else r"$\overline{p}$  "+label_end)
    ax4.plot(times, pressures_rfm, color=deltap_colors[i], linestyle=linestyles[i], 
            marker=markers[i], markevery=me, label=r"$\overline{p}$ right LV$\rightarrow$3V foramen " if i==0 else None)

# Configure plots
ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='k')
ax_.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='k')
ax.set_ylabel('ml/s', labelpad=0)
ax_.set_ylabel('mmHg/cm', labelpad=0)
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
    fig.savefig(fig_dir+f"compare_models_aqueduct_flowrate_pressure_gradients_p={p}_k={k}_mesh={mesh_suffix}_solver={solver_type}")
    fig2.savefig(fig_dir+f"compare_models_flowrates_p={p}_k={k}_mesh={mesh_suffix}_solver={solver_type}")
    fig3.savefig(fig_dir+f"compare_models_flow_volumes_p={p}_k={k}_mesh={mesh_suffix}_solver={solver_type}")
    fig4.savefig(fig_dir+f"compare_models_pressures_p={p}_k={k}_mesh={mesh_suffix}_solver={solver_type}")
        
print("Post processing complete.")

plt.show()