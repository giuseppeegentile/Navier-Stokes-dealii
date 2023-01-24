#ifndef STOKES_HPP
#define STOKES_HPP

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>
 
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
 
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
 
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
 
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
 
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
 
#include <deal.II/numerics/solution_transfer.h>
 
#include <deal.II/lac/sparse_direct.h>
 
#include <deal.II/lac/sparse_ilu.h>

#include <fstream>


/****/
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>

using namespace dealii;
static constexpr unsigned int dim = 2;
static constexpr double re_boundary = 1000.0;
static constexpr double step_size = 2000.0;
static constexpr double tol = 1e-6;
static constexpr int max_line = 15;

class StationaryNavierStokes
{
public:
  StationaryNavierStokes(const unsigned int degree)  : viscosity(1.0 / 7500.0)
                                                      , gamma(1.0)
                                                      , degree(degree)
                                                      , fe(FE_SimplexP<dim>(degree + 1), dim, FE_SimplexP<dim>(degree), 1)
                                                      , quadrature_formula(degree + 2){};
  void run(const unsigned int refinement);


protected:
  void compute_initial_guess();

  double                               viscosity;
  double                               gamma;
  const unsigned int                   degree;
  std::vector<types::global_dof_index> dofs_per_block;


  FESystem<dim>      fe;
  DoFHandler<dim>    dof_handler;

  //set all boundaries to zero, contranint the updates: never update boundary values -> update vector zero
  AffineConstraints<double> zero_constraints;
  //dirichlet bcs on the solution vector
  AffineConstraints<double> nonzero_constraints;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;
  SparseMatrix<double>      pressure_mass_matrix;

  BlockVector<double> present_solution;
  BlockVector<double> newton_update;
  BlockVector<double> system_rhs;
  BlockVector<double> evaluation_point;
  Triangulation<dim> mesh;

  QGauss<dim> quadrature_formula;


private:
  void setup_dofs();

  void initialize_system();

  void assemble(const bool initial_step, const bool assemble_matrix);

  void assemble_system(const bool initial_step);

  void assemble_rhs(const bool initial_step);

  void solve(const bool initial_step);

  void refine_mesh();

  void process_solution(unsigned int refinement);

  void output_results(const unsigned int refinement_cycle) const;

  void newton_iteration(const double       tolerance,
                        const unsigned int max_n_line_searches,
                        const unsigned int max_n_refinements,
                        const bool         is_initial_step,
                        const bool         output_result);


};


//set the velocity on the upper surface of the cavity to one, and zero on the other wall
class BoundaryValues : public Function<dim> {
public:
  BoundaryValues() : Function<dim>(dim + 1) {} //we have to set also the pressure even if we don't contraint it
  virtual double value(const Point<dim> & p, const unsigned int component) const override{
    Assert(component < this->n_components, ExcIndexRange(component, 0, this->n_components));
    if (component == 0 && std::abs(p[dim - 1] - 1.0) < 1e-10)
      return 1.0;

    return 0;
  }
};
 
/*******/
//rhs function is zero: no rhs function
/*******/


//Schur complement preconditioner decomposed as product of 3 matrices: A^-1 to solve Ax = b, solve this with direct solver for now
                                                                    //second factor is a simple matrix vector multiplication
                                                                    //Schur can be approximated by pressure mass and its inverse obtained through an inexact solver
                                                                                      //since pressure mass is SPD->use CG to solve
template <class PreconditionerMp>
class BlockSchurPreconditioner : public Subscriptor {
  public:
    BlockSchurPreconditioner( double                           gamma,
                              double                           viscosity,
                              const BlockSparseMatrix<double> &S,
                              const SparseMatrix<double> &     P,
                              const PreconditionerMp &         Mppreconditioner);

    void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;

  private:
    const double                     gamma;
    const double                     viscosity;
    const BlockSparseMatrix<double> &stokes_matrix;
    const SparseMatrix<double> &     pressure_mass_matrix;
    const PreconditionerMp &         mp_preconditioner;
    SparseDirectUMFPACK              A_inverse;
};


template <class PreconditionerMp>
BlockSchurPreconditioner<PreconditionerMp>::BlockSchurPreconditioner(
  double                           gamma,
  double                           viscosity,
  const BlockSparseMatrix<double> &S,
  const SparseMatrix<double> &     P,
  const PreconditionerMp &         Mppreconditioner)
  : gamma(gamma)
  , viscosity(viscosity)
  , stokes_matrix(S)
  , pressure_mass_matrix(P)
  , mp_preconditioner(Mppreconditioner) {
      A_inverse.initialize(stokes_matrix.block(0, 0));
  }

template <class PreconditionerMp>
void BlockSchurPreconditioner<PreconditionerMp>::vmult(
  BlockVector<double> &      dst,
  const BlockVector<double> &src) const {
    Vector<double> utmp(src.block(0));
    {
      SolverControl solver_control(1000, 1e-6 * src.block(1).l2_norm());
      SolverCG<Vector<double>> cg(solver_control);

      dst.block(1) = 0.0;
      cg.solve(pressure_mass_matrix,
                dst.block(1),
                src.block(1),
                mp_preconditioner);
      dst.block(1) *= -(viscosity + gamma);
    }

    {
      stokes_matrix.block(0, 1).vmult(utmp, dst.block(1));
      utmp *= -1.0;
      utmp += src.block(0);
    }

    A_inverse.vmult(dst.block(0), utmp);
  }
#endif