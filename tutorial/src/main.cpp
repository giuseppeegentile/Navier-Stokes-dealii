#include "NavierStokesSolver.hpp"

// Main function.
int
main()
{
  
      StationaryNavierStokes flow(/* degree = */ 1);
      flow.run(4);

  return 0;
}