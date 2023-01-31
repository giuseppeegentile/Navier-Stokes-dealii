#include "StokesTD.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;

  const double T      = 1.0;
  const double deltat = 0.05;
  const double theta  = 0.0;

  StokesTD problem(degree_velocity, degree_pressure, T, deltat, theta);

  problem.setup();
  problem.solve();

  return 0;
}