import numpy as np
import mpi4py
import pyvista
import dolfinx
import basix.ufl
import colormaps
import adios4dolfinx

mesh_prefix = "medium"
T = 3.0 # Final simulation time
dt = 0.001 # Timestep size
k = 1 # Element degree 
model_versions = ["deformation+cilia+production", "deformation+production"]
filenames = [
    f"../output/ex3/{mesh_prefix}-mesh/flow/navier-stokes/checkpoints/BDM_{model_version}_velocity_T={T}_dt={dt}" \
        for model_version in model_versions
    ]

# Prepare plotting
# Slice coordinates
origin_yz = [-0.000342, 0.0029, 0.0075]
origin_xz = [0.0, -0.0112, 0.0075]
origin_xy = [0.0, 0.0029, 0.0065]

zoom_yz = 2.0
zoom_xz = 1.0
zoom_xy = 1.35

# Colors
from matplotlib import colormaps as cm
# forward_cmap = cm.get_cmap("viridis")
inferno = cm.get_cmap("inferno")
forward_cmap = colormaps.cet_l_blue
backward_cmap = colormaps.cet_l_kry
n_colors = 16
sargs = sargs = {
    'title': 'Velocity magnitude [m/s]',
    'n_labels': 4, 
    'fmt': '%.2g',
    'font_family': 'arial'
}

pl = pyvista.Plotter(shape=(4, 2), window_size=[1000, 1200], border=False)

m = 0 # Column index

view = 'yz'

times = [200, 400, 600, 800]
for i, filename in enumerate(filenames):
    # Prepare finite elements and pyvista grid used to plot
    mesh = adios4dolfinx.read_mesh(filename, comm=mpi4py.MPI.COMM_WORLD)
    BDM = dolfinx.fem.functionspace(mesh,
            basix.ufl.element("BDM", mesh.basix_cell(), k)
            )
    DG = dolfinx.fem.functionspace(mesh,
            basix.ufl.element("DG", mesh.basix_cell(), k, shape=(mesh.geometry.dim,))
            )
    u_bdm = dolfinx.fem.Function(BDM)
    u_dg = dolfinx.fem.Function(DG)
    cells, types, x = dolfinx.plot.vtk_mesh(DG)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    
    for j, time in enumerate(times):
        # Read velocity field and interpolate
        adios4dolfinx.read_function(filename, u_bdm, time=time, name="relative_velocity")
        u_dg.interpolate(u_bdm)
        u_vector = np.vstack(
            (
                u_dg.sub(0).collapse().x.array.copy(),
                u_dg.sub(1).collapse().x.array.copy(),
                u_dg.sub(2).collapse().x.array.copy()
            )
        ).T
        # Average the discontinuous data over each cell
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    
        # Reshape to (num_cells, nodes_per_cell, 3) where nodes_per_cell=4 for a tetrahedron
        u_vector_reshaped = u_vector.reshape(num_cells, 4, 3)
        u_vector_avg = np.mean(u_vector_reshaped, axis=1)
        
        # Set data
        grid.cell_data["u"] = u_vector_avg
        grid.set_active_vectors("u")

        sargs['title'] = f'Time: {time}, v: {model_versions[i]}' # Give each bar a unique title
        # Plot yz plane
        if view=='yz':
            pl.subplot(j, m)
            sliced_grid = grid.slice(normal=[-1, 0, 0], origin=origin_yz)
            pl.add_mesh(sliced_grid,
                        cmap=inferno,
                        # clim=[0, 0.75] if location=="laterals" else [0, 1.0],
                        show_scalar_bar=True,
                        scalar_bar_args=sargs.copy(),
                        n_colors=n_colors)
            pl.add_mesh(sliced_grid.glyph(
                        orient="u",  # Orient by our vectors
                        factor=0.75,      # Control arrow length
                        scale="u"    # Color arrows by vector magnitude
                    ),
                    scalar_bar_args=sargs.copy(),
                    color="white"
                    )
            pl.view_yz(negative=True)
            pl.camera.zoom(zoom_yz)
        elif view=='xz':
            # Plot xz plane
            pl.subplot(j, m)
            pl.add_mesh(grid.slice(normal=[0, 1, 0], origin=origin_xz),
            #-0.00374 further back
                            cmap=inferno,
                            show_scalar_bar=True,
                            scalar_bar_args=sargs.copy(),
                        n_colors=n_colors)
            pl.view_xz(negative=True)
            pl.camera.zoom(zoom_xz)
        elif view=='xy':
            # Plot xy plane
            pl.subplot(j, m)
            pl.add_mesh(grid.slice(normal=[0, 0, 1], origin=origin_xy),
                            cmap=inferno,
                            # clim=[0, 0.75] if location=="laterals" else [0, 1.0],
                            show_scalar_bar=True,
                            scalar_bar_args=sargs.copy(),
                        n_colors=n_colors)
            pl.view_xy(negative=True)
            pl.camera.zoom(zoom_xy)

    m += 1 # Increment column index

save_figs = 0
if save_figs:
    model_string = ""
    for model_version in model_versions:
        model_string += model_version + "+"
    pl.show(screenshot=f"../output/illustrations/velocity-slices-{model_string}.png")
else:
    pl.show()