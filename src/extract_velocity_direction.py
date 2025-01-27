import dolfinx as dfx
import adios4dolfinx as a4d

from mpi4py import MPI

comm = MPI.COMM_WORLD
mesh_prefix = 'medium'
cpoint_filename = f'../output/checkpoints/deforming-mesh-{mesh_prefix}/BDM_cilia_velocity'
