#include "NavStokes.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  std::string output_name="nav-stokes";

  const unsigned int N               = 4;
  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;

  Stokes problem(N, degree_velocity, degree_pressure);

  problem.setup();
  problem.solve_newton();
  problem.output(output_name);

  return 0;
}