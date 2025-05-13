import pyvista
import colormaps

pl = pyvista.Plotter()

mesh_prefix = "medium"
solver_type = "navier-stokes"
initial_time = 100
prefix = f"~/flowVC/output/ex3/brain/{mesh_prefix}-mesh/{solver_type}/BDM_deforming_velocity"
data_filename = f"{prefix}/vFTLE-forward-third-T=20-freq=4.{initial_time}"
vtk_suffix = ".vtk"

data = pyvista.read(data_filename+vtk_suffix)
data = data.threshold(0.0) # Remove all points with values below zero

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
sargs = dict(
    italic=False
)
cmap = colormaps.fall


pl.add_mesh(data.slice(normal=[-1, 0, 0]), cmap=cmap, clim=[0, 0.25], below_color='black', scalar_bar_args=sargs)
pl.view_yz(negative=True)
pl.show()