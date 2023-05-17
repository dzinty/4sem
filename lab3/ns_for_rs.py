from fenics import *
from mshr import *

T = 5  # final time
num_steps = 500  # number of time steps
dt = T / num_steps  # time step size
mu = 0.01  # dynamic viscosity
rho = 1  # density

# Create mesh
channel = Polygon(
    [Point(1, 0), Point(2, 0), Point(2, 1), Point(2, 1), Point(3, 1), Point(3, 2), Point(2, 2), Point(2, 2),
     Point(2, 3), Point(1, 3), Point(1, 2), Point(1, 2), Point(0, 2), Point(0, 1), Point(1, 1), Point(1, 1)])
obstacle = Rectangle(Point(1.3, 1.3), Point(1.7, 1.7))
domain = channel - obstacle
mesh = generate_mesh(domain, 64)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundaries
inflow1 = 'near(x[0], 0.) && between(x[1], std::pair<double, double>(1., 2.))'
inflow2 = 'near(x[0], 3.) && between(x[1], std::pair<double, double>(1., 2.))'
outflow = 'near(x[1], 0.) && between(x[0], std::pair<double, double>(1., 2.)) || ' + \
          'near(x[1], 3.) && between(x[0], std::pair<double, double>(1., 2.))'
walls = '(near(x[1], 1.) && !between(x[0], std::pair<double, double>(1., 2.))) || ' + \
        '(near(x[1], 2.) && !between(x[0], std::pair<double, double>(1., 2.))) || ' + \
        '(near(x[0], 1.) && !between(x[1], std::pair<double, double>(1., 2.))) || ' + \
        '(near(x[0], 2.) && !between(x[1], std::pair<double, double>(1., 2.)))'
rect = 'on_boundary && x[0]>1.1 && x[0]<1.9 && x[1]>1.1 && x[1]<1.9'

# Define inflow profile
inflow_profile1 = Expression(('(x[1] - 1)*(2 - x[1])*(1 + sin(t/0.1))', '0'), degree=2, t=0.)
inflow_profile2 = Expression(('-(x[1] - 1)*(2 - x[1])*(1 + sin(t/0.1))', '0'), degree=2, t=0.)

# Define boundary conditions
bcu_inflow1 = DirichletBC(V, inflow_profile1, inflow1)
bcu_inflow2 = DirichletBC(V, inflow_profile2, inflow2)
bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_rect = DirichletBC(V, Constant((0, 0)), rect)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_inflow1, bcu_inflow2, bcu_walls, bcu_rect]
bcp = [bcp_outflow]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_ = Function(V)
p_n = Function(Q)
p_ = Function(Q)

# Define expressions used in variational forms
U = 0.5 * (u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0))
k = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)


# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))


# Define stress tensor
def sigma(u, p):
    return 2 * mu * epsilon(u) - p * Identity(len(u))


# Define variational problem for step 1
F1 = rho * dot((u - u_n) / k, v) * dx \
     + rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx \
     + inner(sigma(U, p_n), epsilon(v)) * dx \
     + dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds \
     - dot(f, v) * dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q)) * dx
L2 = dot(nabla_grad(p_n), nabla_grad(q)) * dx - (1 / k) * div(u_) * q * dx

# Define variational problem for step 3
a3 = dot(u, v) * dx
L3 = dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# VTK files for visualization
file_u = File('ns_for_rs/velocity.pvd')
file_p = File('ns_for_rs/pressure.pvd')

# Create time series (for use in reaction_system.py)
timeseries_u = TimeSeries('ns_for_rs/velocity_series')
timeseries_p = TimeSeries('ns_for_rs/pressure_series')

# Save mesh to file (for use in reaction_system.py)
File('ns_for_rs/cross.xml.gz') << mesh

# Time-stepping
t = 0
for n in range(num_steps):
    # Update current time
    t += dt
    inflow_profile1.t = t
    inflow_profile2.t = t

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'petsc_amg')

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, 'bicgstab', 'petsc_amg')

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor')

    # Save solution to file (XDMF/HDF5)
    file_u << u_
    file_p << p_

    # Save nodal values to file
    timeseries_u.store(u_.vector(), t)
    timeseries_p.store(p_.vector(), t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

    print("Current time: %f / %f" % (t, T))
