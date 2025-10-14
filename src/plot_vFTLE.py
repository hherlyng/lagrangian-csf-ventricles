import sys
import vtk
import pyvista
import colormaps

location = sys.argv[1]
solver_type = sys.argv[2]
mesh_prefix = sys.argv[3]
mesh_suffixes = {"medium" : 0,
                 "fine" : 1,
                 "very_fine" : 2
}
p = 4
k = 2
model_versions = {1 : "deformation",
                  2 : "deformation+cilia",
                  3 : "deformation+production",
                  4 : "deformation+cilia+production"
}
model_version = model_versions[int(sys.argv[4])]
prefix = f"~/flowVC/output/brain/{mesh_prefix}-mesh/{solver_type}/p={p}_k={k}/{model_version}/vFTLE-vtk"
vtk_suffix = ".vtk"
fig_output_dir = "../output/illustrations/FTLE/"
T = 2
freq = 4

sargs = dict(
    italic=False,
    n_labels=4,
    position_x=0.25,
    position_y=0.80
)

forward_cmap = colormaps.cet_l_blue
forward_cmap = colormaps.viola
backward_cmap = colormaps.cet_l_kry
backward_cmap = colormaps.curl
n_colors = 16
times = [125, 250, 375, 500]
directions = ["forward", "backward"]
if location=="third":
    pl2_shape = [400, 900]
else:
    pl2_shape = [500, 1250]
pl  = pyvista.Plotter(shape=(len(times), len(directions)), window_size=[750, 2000], border=False, off_screen=True, image_scale=2) # yz plane
pl2 = pyvista.Plotter(shape=(len(times), len(directions)), window_size=pl2_shape, border=False, off_screen=True, image_scale=2) # xz plane
pl3 = pyvista.Plotter(shape=(len(times), len(directions)), window_size=[750, 2500], border=False, off_screen=True, image_scale=2) # xy plane

for j, direction in enumerate(directions):
    sargs["title"] = direction
    for i, time in enumerate(times):
        print(f"Time = {time}")
        data_filename = f"{prefix}/vFTLE-{direction}-{location}-T={T}-freq={freq}.{time}" #/{direction}-{location}
        data = pyvista.read(data_filename+vtk_suffix)
        data = data.threshold(0.0) # Remove all points with values below zero

        if location=="third":
            origin_yz = [-0.00156, 0.0029, 0.0075]
            origin_xz = [0.0, -0.0112, 0.0075]
            origin_xy = [0.0, 0.0029, 0.0065]

            zoom_yz = 2.75
            zoom_xz = 1.5
            zoom_xy = 1.35

            clims = [[0, 5.0], # yz
                     [0, 5.0], # xz
                     [0, 5.0], #xy
                     ]

        elif location=="laterals":
            origin_yz = None
            origin_xz = [0.0, 0.01435, 0.0195]
            origin_xy = [0.0, 0.01435, 0.02084]

            zoom_yz = 1.35
            zoom_xz = 3.5
            zoom_xy = 1.25
        
            clims = [[0, 1.0], # yz
                     [0, 2.5], # xz
                     [0, 1.0], #xy
                     ]
        
        else:
            origin_yz = None
            origin_xz = None
            origin_xy = None

            zoom_yz = 1.25
            zoom_xz = 1.25
            zoom_xy = 1.25
        
        # Plot yz plane
        pl.subplot(i, j)
        pl.add_mesh(data.slice(normal=[-1, 0, 0], origin=origin_yz),
                    cmap=forward_cmap if direction=="forward" else backward_cmap,
                    clim=clims[0],
                    show_scalar_bar=False,
                    n_colors=n_colors)
        pl.view_yz(negative=True)
        if location=="third":
            pos = pl.camera.position
            foc = pl.camera.focal_point
            dy = -0.010
            dz = 0.005
            pl.camera.position = (pos[0], pos[1] + dy, pos[2] + dz)
            pl.camera.focal_point = (foc[0], foc[1] + dy, foc[2] + dz)
        pl.camera.zoom(zoom_yz)

        # Plot xz plane
        pl2.subplot(i, j)
        pl2.add_mesh(data.slice(normal=[0, 1, 0], origin=origin_xz),
                     cmap=forward_cmap if direction=="forward" else backward_cmap,
                     clim=clims[1],
                     show_scalar_bar=False,
                     n_colors=n_colors)
        pl2.view_xz(negative=True)
        if location=="laterals":
            pos = pl2.camera.position
            foc = pl2.camera.focal_point
            dz = -0.0025
            dy = 0.0
            pl2.camera.position = (pos[0], pos[1], pos[2] + dz)
            pl2.camera.focal_point = (foc[0], foc[1], foc[2] + dz)
        pl2.camera.zoom(zoom_xz)
        
        # Plot xy plane
        pl3.subplot(i, j)
        pl3.add_mesh(data.slice(normal=[0, 0, 1], origin=origin_xy),
                     cmap=forward_cmap if direction=="forward" else backward_cmap,
                     clim=clims[2],
                     show_scalar_bar=False,
                     n_colors=n_colors)
        pl3.view_xy(negative=True)
        pl3.camera.roll = 90
        pl3.camera.zoom(zoom_xy)
        
pl.screenshot(fig_output_dir+f"vFTLE-mesh_{mesh_suffixes[mesh_prefix]}-slices-human-{location}-{solver_type}-model-{model_version}-yz.png", scale=2)
pl2.screenshot(fig_output_dir+f"vFTLE-mesh_{mesh_suffixes[mesh_prefix]}-slices-human-{location}-{solver_type}-model-{model_version}-xz.png")
pl3.screenshot(fig_output_dir+f"vFTLE-mesh_{mesh_suffixes[mesh_prefix]}-slices-human-{location}-{solver_type}-model-{model_version}-xy.png")
