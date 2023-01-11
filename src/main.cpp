// #include "NavierStokesSolver.hpp"

// // Main function.
// int
// main(int argc, char *argv[])
// {
//   Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

//   const unsigned int degree_velocity = 2;
//   const unsigned int degree_pressure = 1;

//   NavierStokesSolver problem(degree_velocity, degree_pressure);

//   problem.setup();
//   for(int i = 0; i < 5; i++) problem.run_newton_loop(i);

//   return 0;
// }

#include "NavierStokesSolver.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;

  Stokes problem(degree_velocity, degree_pressure);

  problem.setup();
  problem.solve_newton();
  problem.output();

  return 0;
}