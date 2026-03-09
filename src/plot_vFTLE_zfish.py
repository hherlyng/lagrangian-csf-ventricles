import sys
import pyvista
import colormaps
pyvista.start_xvfb()
mesh_prefix = "medium"
solver_type = sys.argv[2]
location = sys.argv[1]
p = 3
k = 1
animal = "zfish"
prefix = "~/flowVC/output/zfish/BDM/vFTLE-vtk"
vtk_suffix = ".vtk"
T = 150
freq = 50


forward_cmap = colormaps.viola
backward_cmap = colormaps.curl
n_colors = 128
clim = [0, 0.1]
times = [30, 60, 90]
pl = pyvista.Plotter(shape=(len(times), 2), window_size=[400, 700], border=False, off_screen=True) # yz plane
pl2 = pyvista.Plotter(shape=(len(times), 2), window_size=[600, 700], border=False, off_screen=True) # xz plane
pl3 = pyvista.Plotter(shape=(2, len(times)), window_size=[700, 350], border=False, off_screen=True) # xy plane
negative = False if animal=="zfish" else True

for j, direction in enumerate(["forward", "backward"]):
    
    for i, time in enumerate(times):
        print(f"Time = {time}")
        data_filename = f"{prefix}/vFTLE-{direction}-{location}-T={T}-freq={freq}.{time}" #/{direction}-{location}
        data = pyvista.read(data_filename+vtk_suffix)
        data = data.threshold(0.0) # Remove all points with values below zero

        origin_yz = None
        origin_xz = None
        origin_xy = None

        zoom_yz = 1.35
        zoom_xz = 1.50
        zoom_xy = 1.55
        
        # Plot yz plane
        pl.subplot(i, j)
        pl.add_mesh(data.slice(normal=[-1, 0, 0], origin=origin_yz),
                    cmap=forward_cmap if direction=="forward" else backward_cmap,
                    clim=clim,
                    show_scalar_bar=False,
                    n_colors=n_colors)
        pl.view_yz(negative=True)
        pl.camera.zoom(zoom_yz)

        # Plot xz plane
        pl2.subplot(i, j)
        pl2.add_mesh(data.slice(normal=[0, 1, 0], origin=origin_xz),
                     cmap=forward_cmap if direction=="forward" else backward_cmap,
                     clim=clim,
                     show_scalar_bar=False,
                     n_colors=n_colors)
        pl2.view_xz(negative=negative)
        pl2.camera.zoom(zoom_xz)
        
        # Plot xy plane
        pl3.subplot(j, i)
        pl3.add_mesh(data.slice(normal=[0, 0, 1], origin=origin_xy),
                     cmap=forward_cmap if direction=="forward" else backward_cmap,
                     clim=clim,
                     show_scalar_bar=False,
                     n_colors=n_colors)
        pl3.view_xy(negative=negative)
        pl3.camera.zoom(zoom_xy)
        
pl.screenshot(f"../output/illustrations/FTLE/vFTLE-slices-{animal}-{location}-{solver_type}-yz.png")
pl2.screenshot(f"../output/illustrations/FTLE/vFTLE-slices-{animal}-{location}-{solver_type}-xz.png")
pl3.screenshot(f"../output/illustrations/FTLE/vFTLE-slices-{animal}-{location}-{solver_type}-xy.png")