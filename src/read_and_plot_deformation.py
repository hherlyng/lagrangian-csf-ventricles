import numpy   as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from pathlib import Path

file_extension = ".svg"
timestep = 0.001
p_refinement_study = False
mesh_refinement_study = True

lw = 3
ms = 12 # Marker size
params = {
    "lines.linewidth" : lw,
    "legend.fontsize" : 35,
    "text.usetex" : True,
    "font.family" : "serif",
    "axes.labelsize" :  40,
    "xtick.labelsize" : 35,
    "ytick.labelsize" : 35,
    "axes.labelpad" : 14.0,
    "legend.frameon" : False
}
plt.rcParams.update(params)

m_to_mm = 1e3 # Meter to millimeter
E = 1500
T = 4
k = 1
linestyles = ["-", "--", "-.", ":"]
colors = ['b', 'g', 'm', 'c']
fig_dir = "../output/illustrations/deformation/"
Path(fig_dir).mkdir(parents=True, exist_ok=True)

fig_kin_en, ax_kin_en = plt.subplots(figsize=([14, 10]))
fig_el_en, ax_el_en = plt.subplots(figsize=([14, 10]))
fig_md, ax_md = plt.subplots(figsize=([14, 10]))
fig_pd, ax_pd = plt.subplots(figsize=([14, 10]))

times = None

if p_refinement_study: 

    mesh_suffix = 0
    degrees = [2, 3, 4, 5]
    
    for j, p in enumerate(degrees):
        data_dir = f"../output/mesh_{mesh_suffix}/deformation_p={p}_E={E:.0f}_k={k}_T={T:.0f}/data/"
        # Prepare data arrays
        energy = np.load(data_dir+f"energies_dt={timestep}.npy", "r")
        max_disp = np.load(data_dir+f"max_displacements_dt={timestep}.npy", "r")
        point_disp = np.load(data_dir+f"point_displacements_dt={timestep}.npy", "r")

        if times is None: times = np.arange(energy.shape[0])*timestep

        ax_kin_en.plot(times, energy[:, 0], color=colors[j], linestyle=linestyles[0], label=rf"$p = {p}$")
        ax_el_en.plot(times, energy[:, 1], color=colors[j], linestyle=linestyles[1], marker='^', markevery=int(len(times)/25), label=rf"$p = {p}$")

        ax_md.plot(times, m_to_mm*max_disp[:, 3], color=colors[j], linestyle=linestyles[0], label=rf"max $|w|$, $p = {p}$" if j==0 else rf"$p = {p}$")

        ax_pd.plot(times, m_to_mm*point_disp[:, 0], color=colors[j], linestyle=linestyles[0], label=rf"$w_x$, $p = {p}$" if j==0 else rf"$p = {p}$")
        ax_pd.plot(times, m_to_mm*point_disp[:, 1], color=colors[j], linestyle=linestyles[1], label=rf"$w_y$" if j==0 else None)
        ax_pd.plot(times, m_to_mm*point_disp[:, 2], color=colors[j], linestyle=linestyles[2], label=rf"$w_z$" if j==0 else None)

    ax_kin_en.legend(loc="upper left")
    ax_el_en.legend(loc="upper left")
    ax_md.legend()
    ax_pd.legend()

    ax_kin_en.set_ylabel("Kinetic energy")
    ax_el_en.set_ylabel("Elastic energy")
    ax_md.set_ylabel("Displacement [mm]")
    ax_pd.set_ylabel("Displacement [mm]")
    [ax.set_xlabel('Time [s]') for ax in [ax_kin_en, ax_el_en, ax_md, ax_pd]]

    plt.show()

    [fig.tight_layout() for fig in [fig_kin_en, fig_el_en, fig_md, fig_pd]]
    fig_kin_en.savefig(fig_dir+f"p_refinement_kinetic_energies_mesh_{mesh_suffix}"+file_extension)
    fig_el_en.savefig(fig_dir+f"p_refinement_elastic_energies_mesh_{mesh_suffix}"+file_extension)
    fig_md.savefig(fig_dir+f"p_refinement_max_displacements_mesh_{mesh_suffix}"+file_extension)
    fig_pd.savefig(fig_dir+f"p_refinement_point_displacements_mesh_{mesh_suffix}"+file_extension)

if mesh_refinement_study:

    p = 3
    mesh_suffixes = [0, 1]

    for j, mesh_suffix in enumerate(mesh_suffixes):
        data_dir = f"../output/mesh_{mesh_suffix}/deformation_p={p}_E={E:.0f}_k={k}_T={T:.0f}/data/"
        # Prepare data arrays
        energy = np.load(data_dir+f"energies_dt={timestep}.npy", "r")
        max_disp = np.load(data_dir+f"max_displacements_dt={timestep}.npy", "r")
        point_disp = np.load(data_dir+f"point_displacements_dt={timestep}.npy", "r")

        if times is None: times = np.arange(energy.shape[0])*timestep

        ax_kin_en.plot(times, energy[:, 0], color=colors[j], linestyle=linestyles[0], label=f"mesh {mesh_suffix+1}")
        ax_el_en.plot(times, energy[:, 1], color=colors[j], linestyle=linestyles[1], marker='^', markevery=int(len(times)/25), label=f"mesh {mesh_suffix+1}")

        ax_md.plot(times, m_to_mm*max_disp[:, 3], color=colors[j], linestyle=linestyles[0], label=rf"max $|w|$, mesh {mesh_suffix+1}" if j==0 else f"mesh {mesh_suffix+1}")

        ax_pd.plot(times, m_to_mm*point_disp[:, 0], color=colors[j], linestyle=linestyles[0], label=rf"$w_x$, mesh {mesh_suffix+1}" if j==0 else f"mesh {mesh_suffix+1}")
        ax_pd.plot(times, m_to_mm*point_disp[:, 1], color=colors[j], linestyle=linestyles[1], label=rf"$w_y$" if j==0 else None)
        ax_pd.plot(times, m_to_mm*point_disp[:, 2], color=colors[j], linestyle=linestyles[2], label=rf"$w_z$" if j==0 else None)

    ax_kin_en.legend(loc="upper left")
    ax_el_en.legend(loc="upper left")
    ax_md.legend()
    ax_pd.legend()

    ax_kin_en.set_ylabel("Kinetic energy")
    ax_el_en.set_ylabel("Elastic energy")
    ax_md.set_ylabel("Displacement [mm]")
    ax_pd.set_ylabel("Displacement [mm]")

    plt.show()

    [fig.tight_layout() for fig in [fig_kin_en, fig_el_en, fig_md, fig_pd]]
    fig_kin_en.savefig(fig_dir+f"mesh_refinement_kinetic_energies_p={p}"+file_extension)
    fig_el_en.savefig(fig_dir+f"mesh_refinement_elastic_energies_p={p}"+file_extension)
    fig_md.savefig(fig_dir+f"mesh_refinement_max_displacements_p={p}"+file_extension)
    fig_pd.savefig(fig_dir+f"mesh_refinement_point_displacements_p={p}"+file_extension, bbox_inches='tight')



# ax_md.plot(times, m_to_mm*max_disp[:, 0], color=colors[j], linestyle=linestyles[0], label=rf"max $w_x$, $p = {p}$" if j==0 else rf"$p = {p}$")
# ax_md.plot(times, m_to_mm*max_disp[:, 1], color=colors[j], linestyle=linestyles[1], label=rf"max $w_y$" if j==0 else None)
# ax_md.plot(times, m_to_mm*max_disp[:, 2], color=colors[j], linestyle=linestyles[2], label=rf"max $w_z$" if j==0 else None)