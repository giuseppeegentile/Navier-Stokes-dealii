The aim of this project is to solve the steady and unsteady Navier-Stokes in 2D equation:

$\rho\frac{\partial{u}}{\partial{t}} + \rho(u  \nabla{u})  + \rho  \nu \bigtriangleup{u} +  \nabla{p} = f $ 

$\ \nabla \cdot u = 0 $

with boundary conditions:

$\ u = u_{in}$      on  $\ \Gamma_{in} $

$\ u = 0$  on $\ \Gamma_{wall}$

$\ \rho \nu (\nabla{u}) n - pn = -\rho_{out} n$    on $\ \Gamma_{out}$

where u and p are the unknowns, respectively the velocity and the pressure, $u_{in}$ is the velocity of the fluid entering our domain,
$\rho_{out}$ is the outlet pressure, $\rho$ is the density of the fluid, and $\nu$ is the kinematic viscosity.

The $\rho\frac{\partial{u}}{\partial{t}}$ term disappears in the steady case.

$\int\limits_\Omega\frac{u^n+1 - u^n}}{\bigtriangleup t}v +\int\limits_\Omega \rho \nu \nabla{u} : \nabla{v}  +\int\limits_\Omega \rho u \nabla{u} v - \int\limits_\Omega \rho u \nabla{v} v = +\int\limits_\Omega fv + \int\limits_\Omega -p_{out}n v$

Our approach was to linearize this system and solve it through the Newton method, using as initial guess the solution of the Stokes problem in the same domain.
Our linearized system reads like this:

$\int\limits_\Omega\frac{\partial{\delta_{u}}}{\bigtriangleup t}v +\int\limits_\Omega \rho \nu \nabla{\delta_{u}} : \nabla{v}  +\int\limits_\Omega \rho u^k \nabla{\delta_{u}} v + \int\limits_\Omega \rho \delta_{u} \nabla{u^k} v + \int\limits_\Omega \delta_{p} \nabla\cdot v = - R(u^k,p^k)(v,q)$

$\ \int\limits_\Omega q \nabla\cdot\delta_{u} = -\int\limits_\Omega q \nabla\cdot u^k $

where $u^k$ and $p^k$ are the values of velocity and pressure at the current step, $\delta_u$ and $ \delta_p$ are the increment of the velocity and pressure guesses, and $v$ and $q$ are the test functions.
$R(u^k,p^k)(v,q)$ is the momentum equation residual in weak formulation, evaluated in $u^k$ and $p^k$:

$R(u^k,p^k)(v,q) =\int\limits_\Omega\frac{\partial{u}}{\partial{t}}v + \int\limits_\Omega \nu \rho \nabla{u^k} : \nabla{v} + \int\limits_\Omega \rho u^k \nabla{u^k} v - \int\limits_\Omega \rho \nabla\cdot v -\int\limits_\Omega f\cdot v - \int\limits_{\Gamma_{out}} -\rho_{out} n v $

The continuity equation residual is as written in the linearized system.

We then assemble a linearized system of the form:

$J_{r} (u,p) \delta = - r(u)$

$J_{r}$ is the jacobian of the residual composed of four elements: the frechet derivative of the first and second equation residual with respect to $u$ and $p$.
\delta is a vector composed of  $\delta_{u}$ and $\delta_{p}$ and $-r(u)$ is the vector of the two equations residual.

The system is solved in order to obtain $\delta$ and use it to update our $u^k$ and $p^k$ until a stop criterion is met.
