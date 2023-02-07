# Time dependent Navier Stokes
Numerical Methods for Partial Differential Equation project @ Politecnico di Milano

By:
  - [Giacomo Carugati](https://github.com/giacomo-carugati)
  - [Filippo Buda](https://github.com/filippobuda)
  - Giuseppe Gentile 
  
## Solver and simulate
Flow past a cylinder simulating a tube with Reynolds number up to 200.
The `deal.ii` library has been used.

The mesh has been written using gmsh and then exported in .msh format.

The available mesh for simulation are in the mesh folder, the output of the simulation is exported as `xdmf` file, therefore can be viewed in paraview.
The code is fully parallelized using `TrilinosWrappers`, so you can run it using mpi.

## Instructions to run the code
Once inside the hpc-courses docker:
  - module load gcc-glibc dealii
  - mkdir build
  - cd build
  - cmake ..
  - make && ./navierstokes
  
  
  ### Weak formulation
  
  The aim of this project is to solve the steady and unsteady Navier-Stokes in 2D equation:

$\rho\frac{\partial{u}}{\partial{t}} + \rho(u  \nabla{u})  + \rho  \nu \bigtriangleup{u} +  \nabla{p} = f $ 

$\ \nabla \cdot u = 0 $

with boundary conditions:

$\ u = u_{in}$      on  $\ \Gamma_{in} $

$\ u = 0$  on $\ \Gamma_{wall}$

$\ \rho \nu (\nabla{u}) n - pn = -\rho_{out} n$    on $\ \Gamma_{out}$

where u and p are the unknowns, respectively the velocity and the pressure, $ u_{in}$ is the velocity of the fluid entering our domain,
$\rho_{out}$ is the outlet pressure, $ \rho$ is the density of the fluid, and $ \nu $ is the kinematic viscosity.

The $\rho\frac{\partial{u}}{\partial{t}}$ term disappears in the steady case.

Our approach was to linearize this system and solve it through the Newton method, using as initial guess the solution of the Stokes problem in the same domain.
Our linearized system reads like this:

$\int\limits_\Omega\frac{\partial{\delta_{u}}}{\bigtriangleup t}v +\int\limits_\Omega \rho \nu \nabla{\delta_{u}} : \nabla{v}  +\int\limits_\Omega \rho u^k \nabla{\delta_{u}} v + \int\limits_\Omega \rho \delta_{u} \nabla{u^k} v + \int\limits_\Omega \delta_{p} \nabla\cdot v = - R(u^k,p^k)(v,q)$

$\ \int\limits_\Omega q \nabla\cdot\delta_{u} = -\int\limits_\Omega q \nabla\cdot u^k $

where $u^k$ and $p^k$ are the values of velocity and pressure at the current step, $ \delta_u $ and $ \delta_p $ are the increment of the velocity and pressure guesses, and $ v $ and $ q $ are the test functions.
$ R(u^k,p^k)(v,q)$ is the momentum equation residual in weak formulation, evaluated in $ u^k $ and $ p^k $:

$R(u^k,p^k)(v,q) =\int\limits_\Omega\frac{\partial{u}}{\partial{t}}v + \int\limits_\Omega \nu \rho \nabla{u^k} : \nabla{v} + \int\limits_\Omega \rho u^k \nabla{u^k} v - \int\limits_\Omega \rho \nabla\cdot v -\int\limits_\Omega f\cdot v - \int\limits_{\Gamma_{out}} -\rho_{out} n v $

The continuity equation residual is as written in the linearized system.

We then assemble a linearized system of the form:

$J_{r} (u,p) \delta = - r(u)$

$J_{r}$ is the jacobian of the residual composed of four elements: the frechet derivative of the first and second equation residual with respect to $ u $ and $ p $.
\delta is a vector composed of  $\delta_{u}$ and $\delta_{p}$ and $-r(u)$ is the vector of the two equations residual.

The system is solved in order to obtain $\delta$ and use it to update our $u^k$ and $p^k$ until a stop criterion is met.
=======
$J_{r} (u^{k,n},p^{k,n}) \delta = - r(u)$

$J_{r}$ is the jacobian of the residual composed of four elements: the frechet derivative of the first and second equation residual with respect to $u$ and $p$.
\delta is a vector composed of  $\delta_{u}$ and $\delta_{p}$ and $-r(u)$ is the vector of the two equations residual.

$\begin{bmatrix}A & B^{T} \\ B & 0\end{bmatrix}\begin{bmatrix} \delta_{u} \\ \delta_{p} \end{bmatrix} = \begin{bmatrix} -R_{1} \\ -R_{2} \end{bmatrix}$

where

$A_{i,j}=\int\limits_\Omega\phi_{i}\frac{\phi_{j}}{\bigtriangleup{t}}+\int\limits_\Omega\rho\nu\nabla{\phi_{i}}\nabla{\phi_{j}} +\int\limits_\Omega\rho\phi_{i}\nabla{u^{k}}\phi_{j} + \int\limits_\Omega\rho u^{k} \nabla{\phi_{i}}\phi_{j}$

$B_{i,j}=-\int\limits_\Omega\psi_{i}\nabla\cdot\phi{j}$

and $R_{1}$ and $R_{2}$ are the residual of the first and second equation
The system is solved in order to obtain $\delta$ and use it to update our $u^k$ and $p^k$ until a stop criterion is met.

