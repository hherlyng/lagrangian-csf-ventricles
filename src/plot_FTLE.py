import pyvista
import colormaps

pl = pyvista.Plotter()

mesh_prefix = "medium"
solver_type = "navier-stokes"
location = "laterals"
prefix = f"../output/ex3/mesh_0/vFTLE-vtk/deformation+production/vFTLE-vtk/backward-laterals"
vtk_suffix = ".vtk"

# Colorbar arguments
# sargs = dict(
#     title_font_size=100,
#     label_font_size=150,
#     shadow=True,
#     n_labels=2,
#     italic=True,
#     fmt="%.1g",
#     font_family="times",
#     vertical=True,
#     title='',
#     width=0.05,
#     height=0.8,
#     position_y=0.1
# )
# pl.camera.position    = (x_mid, 0, z_mid*1.1)
# pl.camera.focal_point = (x_mid, y_max, z_mid)
sargs = dict(
    italic=False
)

# from matplotlib import colormaps as cm

# forward_cmap = cm.get_cmap("viridis")
# backward_cmap = cm.get_cmap("inferno")
forward_cmap = colormaps.cet_l_blue
backward_cmap = colormaps.cet_l_kry
n_colors = 12
T = 2
freq = 4
clim1 = [0, 12.5]
clim2 = [0, 2.5]
clim3 = [0, 15]
times = [50, 125, 200]
pl = pyvista.Plotter(shape=(len(times), 1), window_size=[700, 700], border=False) # yz plane
pl2 = pyvista.Plotter(shape=(len(times), 1), window_size=[500, 800], border=False) # xz plane
pl3 = pyvista.Plotter(shape=(len(times), 1), window_size=[500, 1000], border=False) # xy plane

camera_position_1 = [(-0.05419887648610337, -0.021395628631013717, -0.009173851836273835),
        (6.158117322387207e-22, -0.021395628631013717, -0.009173851836273835),
        (0.0, 0.0, 1.0)]

for j, direction in enumerate(["backward"]):
    for i, time in enumerate(times):
        data_filename = f"{prefix}/vFTLE-{direction}-{location}-T={T}-freq={freq}.{time}"
        data = pyvista.read(data_filename+vtk_suffix)
        data = data.threshold(0.0) # Remove all points with values below zero

        # if location=="third":
        origin_yz = [0.0, -0.02, -0.005]
        origin_xz = [-0.0015, -0.0275, -0.0275]
        origin_xy = [-0.0015, -0.015, -0.005]

        zoom_yz = 1.5
        zoom_xz = 1.0
        zoom_xy = 1.35
        # else:
        #     origin_yz = None
        #     origin_xz = [0.0, 0.01435, 0.0195]
        #     origin_xy = [0.0, 0.01435, 0.02084]

        #     zoom_yz = 1.35
        #     zoom_xz = 1.25 
        #     zoom_xy = 1.25
        
        # Plot yz plane
        pl.subplot(i, j)
        # pl.add_mesh(data.slice(normal=[-1, 0, 0], origin=origin_yz),
        pl.add_mesh(data.contour(isosurfaces=[2, 4, 6, 8, 10, 12, 14]),
                    cmap=forward_cmap if direction=="forward" else backward_cmap,
                    clim=clim1,
                    show_scalar_bar=False,
                    n_colors=n_colors)
        pl.view_yz(negative=True)
        pl.camera_position = camera_position_1
        # pl.camera.zoom(zoom_yz)

        # Plot xz plane
        pl2.subplot(i, j)
        pl2.add_mesh(data.slice(normal=[0, 1, 0], origin=origin_xz),
                     cmap=forward_cmap if direction=="forward" else backward_cmap,
                     clim=clim2,
                     show_scalar_bar=False,
                    n_colors=n_colors)
        pl2.view_xz(negative=True)
        pl2.camera.zoom(zoom_xz)
        
        # Plot xy plane
        pl3.subplot(i, j)
        pl3.add_mesh(data.slice(normal=[0, 0, 1], origin=origin_xy),
                     cmap=forward_cmap if direction=="forward" else backward_cmap,
                     clim=clim3,
                     show_scalar_bar=False,
                    n_colors=n_colors)
        pl3.view_xy(negative=True)
        pl3.camera.zoom(zoom_xy)

save_figs = 0
if save_figs:
    pl.show(screenshot=f"../output/illustrations/vFTLE-slices-{location}-{solver_type}-yz.png")
    pl2.show(screenshot=f"../output/illustrations/vFTLE-slices-{location}-{solver_type}-xz.png")
    pl3.show(screenshot=f"../output/illustrations/vFTLE-slices-{location}-{solver_type}-xy.png")
else:
    pl.show()
    pl2.show()
    pl3.show()

# Save camera positions and print
saved_camera_position_1 = pl.camera_position
saved_camera_position_2 = pl2.camera_position
saved_camera_position_3 = pl3.camera_position

print(f"Saved camera position 1: {saved_camera_position_1}")
print(f"Saved camera position 2: {saved_camera_position_2}")
print(f"Saved camera position 3: {saved_camera_position_3}")