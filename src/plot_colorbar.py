import sys
import matplotlib
import matplotlib.pyplot as plt
import colormaps

# Set matplotlib properties
plt.rcParams.update({
    "font.family" : "serif",
    "font.serif" : ["Times New Roman"],
    "text.usetex" : True,
    "text.latex.preamble": r"\usepackage{mathptmx}"
})

if __name__=='__main__':
    # Create and plot a colorbar with the selected colormap
    
    orientation = sys.argv[1]
    cmap_name = sys.argv[2]
    n_colors = int(sys.argv[3])

    if orientation=="horizontal":
        fig = plt.figure(figsize=(8, 1.75))
        ax = fig.add_axes([0.1, 0.35, 0.8, 0.2])
        try:
            cmap = getattr(colormaps, cmap_name)
        except:
            cmap = matplotlib.colormaps.get_cmap(cmap_name)
        
        cmap = cmap.resampled(n_colors)
        cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, orientation='horizontal')
        cb_max = 0.1 # 26.4
        # cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cb.set_ticks([0, 1.0])
        # cb.set_ticklabels([0,
        #                    f"{0.25*cb_max:.2g}",
        #                    f"{0.50*cb_max:.2g}",
        #                    f"{0.75*cb_max:.2g}",
        #                    f"{cb_max:.3g}"])
        cb.set_ticklabels(["Low", "High"])
        ax.tick_params(axis='x', labelsize=35)
        ax.set_title(r"Backward FTLE value", fontsize=46, pad=20.0)
        # ax.set_title(r"Velocity [$\mu$m/s]", fontsize=46, pad=20.0)
        fig_name = f"../output/illustrations/colorbars/{orientation}_colorbar_{cmap_name}_low_high.png"

    fig.tight_layout(pad=0)
    fig.savefig(fig_name, bbox_inches=[])
    plt.show()