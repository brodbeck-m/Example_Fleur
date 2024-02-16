# --- Imports ---
from mpi4py import MPI
import numpy as np
import typing

import basix
import ufl

import dolfinx.fem as dfem
import dolfinx.fem.petsc as dfem_petsc
import dolfinx.mesh as dmesh


def interpolate_ufl_to_function(f_ufl: typing.Any, f: dfem.Function):
    # Create expression
    expr = dfem.Expression(f_ufl, f.function_space.element.interpolation_points())

    # Perform interpolation
    f.interpolate(expr)


# --- Input
nelmt = 2

# --- Create mesh
# The mesh
domain = dmesh.create_unit_square(MPI.COMM_WORLD, nelmt, nelmt, dmesh.CellType.triangle)

# The boundary facets
domain.topology.create_connectivity(1, 2)
boundary_facets = dmesh.exterior_facet_indices(domain.topology)

# --- Setup problem
# Element definition
V = basix.ufl.element("RT", domain.basix_cell(), 1)
P = basix.ufl.element("Lagrange", domain.basix_cell(), 1)

W = dfem.functionspace(domain, basix.ufl.mixed_element([V, P]))

s, u = ufl.TrialFunctions(W)
t, v = ufl.TestFunctions(W)

# Weak form
x = ufl.SpatialCoordinate(domain)

f = 4.0
uex = -0.5 + x[0] * x[0] + x[1] * x[1]

a = ufl.inner(s, t) * ufl.dx + ufl.div(s) * ufl.div(t) * ufl.dx
a += ufl.inner(ufl.grad(u), t) * ufl.dx + ufl.inner(ufl.grad(v), s) * ufl.dx
a += ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

l = ufl.div(t) * ufl.dx

# Boundary conditions
list_bcs = []

V_u, _ = W.sub(1).collapse()
uD = dfem.Function(V_u)

uD.x.array[:] = 1.0  # Set values to 1 (otherwise use interpolate_ufl_to_function)

dofs = dfem.locate_dofs_topological((W.sub(1), V_u), 1, boundary_facets)
list_bcs.append(dfem.dirichletbc(uD, dofs, W.sub(1)))

# Solve problem
problem = dfem_petsc.LinearProblem(
    a, l, bcs=list_bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
uh = problem.solve()


# Error calculation
W_ext = dfem.functionspace(
    domain, basix.ufl.element("Lagrange", domain.basix_cell(), 2)
)
u_ext = dfem.Function(W_ext)

interpolate_ufl_to_function(uex, u_ext)

uh_u = uh.sub(1).collapse()
error_local = dfem.assemble_scalar(
    dfem.form(ufl.inner(uh_u - u_ext, uh_u - u_ext) * ufl.dx)
)

print(np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM)))
