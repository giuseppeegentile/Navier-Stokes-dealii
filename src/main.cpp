#include "NSSteady.hpp"
#include "NSUnsteady.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  std::string output_name="nav-stokes";

  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;

  const double T      = 4.0;
  const double deltat = 0.05;

  // NSSteady problem(N, degree_velocity, degree_pressure);

  // problem.setup();
  // problem.solve_newton();
  // problem.output(output_name);

  NSUnsteady problem(degree_velocity,degree_pressure,T,deltat);

  problem.setup();
  problem.solve();

  return 0;
}