import pyvista
import colormaps

pl = pyvista.Plotter()

mesh_prefix = "medium"
solver_type = "navier-stokes"
time = 500
prefix = f"~/flowVC/output/ex3/brain/{mesh_prefix}-mesh/{solver_type}/BDM_deforming_velocity"
data_filename = f"{prefix}/vFTLE-forward-third-T=10-freq=10.{time}"
vtk_suffix = ".vtk"

data = pyvista.read(data_filename+vtk_suffix)
data = data.threshold(0.0) # Remove all points with values below zero

solver_type2 = "stokes"
prefix2 = f"~/flowVC/output/ex3/brain/{mesh_prefix}-mesh/{solver_type2}/BDM_deforming_velocity"
data_filename2 = f"{prefix2}/vFTLE-forward-third-T=10-freq=10.{time}"
data2 = pyvista.read(data_filename2+vtk_suffix)
data2 = data2.threshold(0.0)

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
cmap = colormaps.fusion

data3 = data.copy()
data3["scalar_field"] = data2["scalar_field"] - data["scalar_field"]

# pl.add_mesh(data3.slice(normal=[-1, 0, 0]), cmap=cmap, clim=[0, 0.25], below_color='black', scalar_bar_args=sargs)
pl.add_mesh(data3.slice(normal=[-1, 0, 0]), cmap=cmap,clim=[-0.5, 0.5], scalar_bar_args=sargs)
pl.view_yz(negative=True)
pl.show()