#ifndef STOKES_HPP
#define STOKES_HPP


#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>
 #include <deal.II/lac/block_vector.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
 #include <deal.II/lac/trilinos_solver.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
 

#include <deal.II/numerics/error_estimator.h>
 
#include <deal.II/numerics/solution_transfer.h>
 
 
/**********************************/
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

 
#include <deal.II/lac/sparse_direct.h>
 
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_gmres.h>

#include <fstream>
#include <iostream>

using namespace dealii;
  
// Class implementing a solver for the Stokes problem.
class NavierStokesSolver
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual void
    vector_value(const Point<dim> & /*p*/,
                 Vector<double> &values) const override
    {
      for (unsigned int i = 0; i < dim - 1; ++i)
        values[i] = 0.0;

      values[dim - 1] = -g;
    }

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      if (component == dim - 1)
        return -g;
      else
        return 0.0;
    }

  protected:
    const double g = 0.0;
  };

  // Function for inlet velocity. This actually returns an object with four
  // components (one for each velocity component, and one for the pressure), but
  // then only the first three are really used (see the component mask when
  // applying boundary conditions at the end of assembly). If we only return
  // three components, however, we may get an error message due to this function
  // being incompatible with the finite element space.
  class InletVelocity : public Function<dim> //from first test case
  {
  public:
    InletVelocity()
      : Function<dim>(dim + 1)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      values[0] = (4 * u_m * p[1] *(H - p[1])) / (H*H);
    //values[0] = 0.0;
      for (unsigned int i = 1; i < dim + 1; ++i)
        values[i] = 0.0;
    }

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      if (component == 0)
        return (4 * u_m * p[1] *(H - p[1])) / (H*H);
      else
        return 0.0;
    }

  protected:
    const double u_m=0.3; // [m/s]
    const double H=0.41; // [m]
  };

  // Since we're working with block matrices, we need to make our own
  // preconditioner class. A preconditioner class can be any class that exposes
  // a vmult method that applies the inverse of the preconditioner.

  // Identity preconditioner.
  class PreconditionIdentity
  {
  public:
    // Application of the preconditioner: we just copy the input vector (src)
    // into the output vector (dst).
    void
    vmult(TrilinosWrappers::MPI::Vector &      dst,
          const TrilinosWrappers::MPI::Vector &src) const
    {
      dst = src;
    }
  protected:
  };

  // Identity block preconditioner.
  class PreconditionBlockIdentity
  {
  public:
    // Application of the preconditioner: we just copy the input vector (src)
    // into the output vector (dst).
    void
    vmult(TrilinosWrappers::MPI::BlockVector &      dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      dst = src;
    }
  protected:
  };

  
  // Block-diagonal preconditioner.
  class PreconditionBlockDiagonal
  {
  public:
    // Initialize the preconditioner, given the velocity stiffness matrix, the
    // pressure mass matrix.
    void
    initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;

      preconditioner_velocity.initialize(velocity_stiffness_);
      preconditioner_pressure.initialize(pressure_mass_);
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector &      dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      ReductionControl  solver_control_velocity(2000, 1e-6 , 1e-2 *src.block(0).l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
        solver_control_velocity);
      solver_cg_velocity.solve(*velocity_stiffness,
                               dst.block(0),
                               src.block(0),
                               preconditioner_velocity);

      ReductionControl   solver_control_pressure(2000, 1e-6, 1e-2 * src.block(1).l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
        solver_control_pressure);
      solver_cg_pressure.solve(*pressure_mass,
                               dst.block(1),
                               src.block(1),
                               preconditioner_pressure);
    }

  protected:
    // Velocity stiffness matrix.
    const TrilinosWrappers::SparseMatrix *velocity_stiffness;

    // Preconditioner used for the velocity block.
    TrilinosWrappers::PreconditionILU preconditioner_velocity;

    // Pressure mass matrix.
    const TrilinosWrappers::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionILU preconditioner_pressure;
  };

  // Block-triangular preconditioner.
  class PreconditionBlockTriangular
  {
  public:
    // Initialize the preconditioner, given the velocity stiffness matrix, the
    // pressure mass matrix.
    void
    initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_,
               const TrilinosWrappers::SparseMatrix &B_)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;
      B                  = &B_;

      preconditioner_velocity.initialize(velocity_stiffness_);
      preconditioner_pressure.initialize(pressure_mass_);
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector &      dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      ReductionControl  solver_control_velocity(1000, 1e-6 , 1e-2 *src.block(0).l2_norm());

      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
        solver_control_velocity);
      solver_cg_velocity.solve(*velocity_stiffness,
                               dst.block(0),
                               src.block(0),
                               preconditioner_velocity);

      tmp.reinit(src.block(1));
      B->vmult(tmp, dst.block(0));
      tmp.sadd(-1.0, src.block(1));

      ReductionControl   solver_control_pressure(1000, 1e-6, 1e-2 * src.block(1).l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
        solver_control_pressure);
      solver_cg_pressure.solve(*pressure_mass,
                               dst.block(1),
                               tmp,
                               preconditioner_pressure);
    }

  protected:
    // Velocity stiffness matrix.
    const TrilinosWrappers::SparseMatrix *velocity_stiffness;

    // Preconditioner used for the velocity block.
    TrilinosWrappers::PreconditionILU preconditioner_velocity;

    // Pressure mass matrix.
    const TrilinosWrappers::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionILU preconditioner_pressure;

    // B matrix.
    const TrilinosWrappers::SparseMatrix *B;

    // Temporary vector.
    mutable TrilinosWrappers::MPI::Vector tmp;
  };


  //===================================================
  class PersonalizedPreconditioner
  {
    public:
      void initialize(const TrilinosWrappers::SparseMatrix &A_/*(0,0) block system matrix*/,
                      const TrilinosWrappers::SparseMatrix &Bt_/*(0,1) block in system_matrix*/,
                      const TrilinosWrappers::SparseMatrix &M_p_, /*pressure mass*/ 
                      const double &nu_,
                      const double &rho_)
      {
        A = &A_;
        Bt = &Bt_;
        M_p = &M_p_;
        nu= &nu_;
        rho=&rho_;
        preconditioner_pressure.initialize(M_p_);
        preconditioner_A.initialize(A_);
      }
    //Application of preconditione<<<<<
    //dst is the result of the preconditoning, src are the normal unknows
    void
    vmult(TrilinosWrappers::MPI::BlockVector &dst,
          const TrilinosWrappers::MPI::BlockVector &src)const
    {
      u_tmp=src.block(0);
      //computing  dst.block(1) that means S^-1 * xp
      SolverControl solver_control_pressure(1000, 1e-2 * src.block(1).l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg_pressure(solver_control_pressure);
      
      dst.block(1) = 0.0;
      solver_cg_pressure.solve(*M_p,
                               dst.block(1),
                               src.block(1),
                               preconditioner_pressure);
      dst.block(1) *= -((*nu * (*rho)) + gamma);
      //computing dst.block(0) that means A^-1 * xu + B^-T * xp
      
      Bt->vmult(u_tmp, dst.block(1));
      u_tmp *= -1.0;
      u_tmp += src.block(0);
      
      SolverControl solver_control_A(100000, 1e-2 * u_tmp.l2_norm());
      //TrilinosWrappers::SolverDirect solver(solver_control_A);
      //PreconditionIdentity preconditioner;
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control_A);
      //SparseDirectUMFPACK solver;
      //solver.initialize(A);
      preconditioner_A.initialize(*A);
      solver.solve(*A, dst.block(0), u_tmp,preconditioner_A);
    }

    private:
    const TrilinosWrappers::SparseMatrix *A;
    const TrilinosWrappers::SparseMatrix *Bt;
    //pressure mass matrix
    const TrilinosWrappers::SparseMatrix *M_p;
    //preconditioner for pressure mass matrix
    mutable TrilinosWrappers::PreconditionILU preconditioner_pressure;
    mutable TrilinosWrappers::PreconditionJacobi preconditioner_A;
    
    const double *nu;
    const double *rho;
    const double gamma=1.0;
    
    //temporary vector
    mutable TrilinosWrappers::MPI::Vector u_tmp;
  };
 //===========================================================================00
  // Constructor.
  NavierStokesSolver(
         const unsigned int &degree_velocity_,
         const unsigned int &degree_pressure_,
         const double &T_,
         const double &deltat_
         )
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , degree_velocity(degree_velocity_)
    , degree_pressure(degree_pressure_)
    , T(T_)
    , deltat(deltat_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Setup system.
  void
  setup();

  // void solve_stokes_system();

  // void assemble_stokes_system();

  // Output results.
  void
  output(const unsigned int &time_step, const double &time) const;

  void 
  solve_newton();

  void solve();
protected:
  // Assemble system. We also assemble the pressure mass matrix (needed for the
  // preconditioner).
  void
  assemble_system();

  // Solve system.
  void
  solve_system();

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // Kinematic viscosity [m2/s].
  const double nu = 0.001;

   // Fluid density.
  const double rho = 1;

  // Outlet pressure [Pa].
  const double p_out = 10;

  // Forcing term.
  ForcingTerm forcing_term;

  // Inlet velocity.
  InletVelocity inlet_velocity;

  // Discretization. ///////////////////////////////////////////////////////////

  // Polynomial degree used for velocity.
  const unsigned int degree_velocity;

  const double T;
  double time;
  const double deltat;

  // Polynomial degree used for pressure.
  const unsigned int degree_pressure;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula for face integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs owned by current process in the velocity and pressure blocks.
  std::vector<IndexSet> block_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // DoFs relevant to current process in the velocity and pressure blocks.
  std::vector<IndexSet> block_relevant_dofs;

  // System matrix.
  TrilinosWrappers::BlockSparseMatrix jacobian_matrix;

  // Pressure mass matrix, needed for preconditioning. We use a block matrix for
  // convenience, but in practice we only look at the pressure-pressure block.
  TrilinosWrappers::BlockSparseMatrix pressure_mass;

  TrilinosWrappers::BlockSparseMatrix stiffness_matrix;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::BlockVector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::BlockVector solution;

  // Increment of the solution between Newton iterations.
  TrilinosWrappers::MPI::BlockVector delta_owned;

  // Residual vector.
  TrilinosWrappers::MPI::BlockVector residual_vector;



  // Right-hand side vector in the Stokes system.
  TrilinosWrappers::MPI::BlockVector stokes_system_rhs;

  // Pressure mass matrix, needed for preconditioning the Stokes system. We use a block matrix for
  // convenience, but in practice we only look at the pressure-pressure block.
  TrilinosWrappers::BlockSparseMatrix stokes_pressure_mass;
    // System solution at previous time step.
  TrilinosWrappers::MPI::BlockVector solution_old;
};

#endif