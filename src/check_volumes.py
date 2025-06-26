from basix.ufl import element
import adios4dolfinx
import mpi4py
import dolfinx
import ufl
import numpy as np
cpoint_filename = "../output/medium-mesh/deformation/checkpoints/displacement_velocity_dt=0.001_T=3"
mesh = adios4dolfinx.read_mesh(cpoint_filename, mpi4py.MPI.COMM_WORLD)
ct = adios4dolfinx.read_meshtags(cpoint_filename, mesh, "ct")
cg2 = element("CG", mesh.basix_cell(), 2, shape=(3,))

V = dolfinx.fem.functionspace(mesh, cg2)
v = dolfinx.fem.Function(V)

r = ufl.SpatialCoordinate(mesh)
chi = r + v
F = ufl.grad(chi)
J = ufl.det(F)

dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)


times = np.arange(1000)

THIRD_VENTRICLE = 4
AQUEDUCT = 5
tags = (THIRD_VENTRICLE, AQUEDUCT)
vol0 = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1*dx(tags)))

changes = []

for t in times:
    adios4dolfinx.read_function(filename=cpoint_filename, u=v, time=t, name="defo_displacement")
    vol = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1*J*dx(tags)))
    print(vol)
    print((vol/vol0-1)*100)
    changes.append(vol-vol0)

import matplotlib.pyplot as plt
plt.plot(times, np.array(changes))
plt.show()