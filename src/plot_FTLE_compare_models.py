import pyvista
import colormaps

pl = pyvista.Plotter()

mesh_prefix = "medium"
location = "laterals"
prefix1 = f"~/flowVC/output/ex3/brain/{mesh_prefix}-mesh/navier-stokes/BDM_deforming_velocity"
prefix2 = f"~/flowVC/output/ex3/brain/{mesh_prefix}-mesh/stokes/BDM_deforming_velocity"
vtk_suffix = ".vtk"

# from matplotlib import colormaps as cm

# forward_cmap = cm.get_cmap("viridis")
# backward_cmap = cm.get_cmap("inferno")
forward_cmap = colormaps.cet_l_blue
backward_cmap = colormaps.cet_l_kry
n_colors = 256

time = 500
pl = pyvista.Plotter(shape=(3, 4), window_size=[700, 700], border=False)

m = 0 # Column index
for direction in ["forward", "backward"]:
    for prefix in [prefix1, prefix2]:
        data_filename = f"{prefix}/vFTLE-{direction}-{location}-T=10-freq=10.{time}"
        data = pyvista.read(data_filename+vtk_suffix)
        data = data.threshold(0.0) # Remove all points with values below zero

        if location=="third":
            origin_yz = [-0.000342, 0.0029, 0.0075]
            origin_xz = [0.0, -0.0112, 0.0075]
            origin_xy = [0.0, 0.0029, 0.0065]

            zoom_yz = 1.5
            zoom_xz = 1.0
            zoom_xy = 1.35
        else:
            origin_yz = None
            origin_xz = [0.0, 0.01435, 0.0195]
            origin_xy = [0.0, 0.01435, 0.02084]

            zoom_yz = 1.15
            zoom_xz = 1.35 
            zoom_xy = 1.35
        
        # Plot yz plane
        pl.subplot(0, m)
        pl.add_mesh(data.slice(normal=[-1, 0, 0], origin=origin_yz),
                    cmap=forward_cmap if direction=="forward" else backward_cmap,
                    clim=[0, 0.75] if location=="laterals" else [0, 1.0],
                    show_scalar_bar=False,
                    n_colors=n_colors)
        pl.view_yz(negative=True)
        pl.camera.zoom(zoom_yz)

        # Plot xz plane
        pl.subplot(1, m)
        pl.add_mesh(data.slice(normal=[0, 1, 0], origin=origin_xz),
        #-0.00374 further back
                     cmap=forward_cmap if direction=="forward" else backward_cmap,
                     clim=[0, 0.75] if location=="laterals" else [0, 1.0],
                     show_scalar_bar=False,
                    n_colors=n_colors)
        pl.view_xz(negative=True)
        pl.camera.zoom(zoom_xz)
        
        # Plot xy plane
        pl.subplot(2, m)
        pl.add_mesh(data.slice(normal=[0, 0, 1], origin=origin_xy),
                     cmap=forward_cmap if direction=="forward" else backward_cmap,
                     clim=[0, 0.75] if location=="laterals" else [0, 1.0],
                     show_scalar_bar=False,
                    n_colors=n_colors)
        pl.view_xy(negative=True)
        pl.camera.zoom(zoom_xy)

        m += 1 # Increment column index

save_figs = 1
if save_figs:
    pl.show(screenshot=f"../output/illustrations/vFTLE-slices-{location}-compare-models.png")
else:
    pl.show()