
#include "NavierStokesSolver.hpp"


void
NavierStokesSolver::setup()
{
  // Create the mesh.
  {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh);

    std::ifstream grid_in_file("../meshmesh-square-h0.100000.msh");

    grid_in.read_msh(grid_in_file);
  }

  // Initialize the finite element space.
  {
    fe         = std::make_unique<FE_SimplexP<dim>>(1);
    quadrature = std::make_unique<QGaussSimplex<dim>>(2);
  }

  // Initialize the DoF handler.
  {
    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // Compute support points for the DoFs.
    FE_SimplexP<dim> fe_linear(1);
    MappingFE        mapping(fe_linear);
  }

  // Initialize the linear system.
  {
    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      {
        for (unsigned int d = 0; d < dim + 1; ++d)
          {
            if (c == dim && d == dim) // pressure-pressure term
              coupling[c][d] = DoFTools::none;
            else // other combinations
              coupling[c][d] = DoFTools::always;
          }
      }

    TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs,
                                                    MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity);
    sparsity.compress();

    // We also build a sparsity pattern for the pressure mass matrix.
    for (unsigned int c = 0; c < dim + 1; ++c)
      {
        for (unsigned int d = 0; d < dim + 1; ++d)
          {
            if (c == dim && d == dim) // pressure-pressure term
              coupling[c][d] = DoFTools::always;
            else // other combinations
              coupling[c][d] = DoFTools::none;
          }
      }
    TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
      block_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    coupling,
                                    sparsity_pressure_mass);
    sparsity_pressure_mass.compress();

    // pcout << "  Initializing the matrices" << std::endl;
    system_matrix.reinit(sparsity);
    pressure_mass.reinit(sparsity_pressure_mass);

    // pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
    // pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
  }
}


void NavierStokesSolver::assemble_system() {

      pcout << "===============================================" << std::endl;
    pcout << "Assembling the system" << std::endl;

    //clear the system matrix and rhs, because these change every iteration
    // system_matrix.reinit(sparsity_pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    const int dofs_per_cell = fe->dofs_per_cell;
    const int n_q_points = quadrature->size();
    const int n_q_points_face = quadrature_face->size();
    std::vector<Tensor<1,dim> > previous_newton_velocity_values (n_q_points);
    std::vector<Tensor< 2, dim> > previous_newton_velocity_gradients (n_q_points);
    // std::vector<Vector<double> > rhs_values (n_q_points, Vector<double>(dim+1));
    std::vector<Tensor<2,dim> > grad_phi_u(dofs_per_cell);
    std::vector<double> div_phi_u(dofs_per_cell);
    std::vector<double> phi_p(dofs_per_cell);
    std::vector<Tensor<1,dim> > phi_u(dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    FEValues<dim> fe_values(*fe, *quadrature, update_values |
                                update_gradients | update_JxW_values | update_quadrature_points);
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(), endc = dof_handler.end();

    for (; cell!=endc; ++cell) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;
        //Calculate velocity values and gradients from previous newton iteration
        //at each quadrature point in cell
        fe_values[velocities].get_function_values(previous_newton_step,
            previous_newton_velocity_values);
        fe_values[velocities].get_function_gradients(previous_newton_step,
            previous_newton_velocity_gradients);
        // forcing_term.vector_value(fe_values.get_quadrature_points(), rhs_values);
        //calculate cell contribution to system
        for (int q = 0; q < n_q_points; q++) {
          Vector<double> forcing_term_loc(dim);
          forcing_term.vector_value(fe_values.quadrature_point(q),
                                    forcing_term_loc);
          Tensor<1, dim> forcing_term_tensor;
          for (unsigned int d = 0; d < dim; ++d)
            forcing_term_tensor[d] = forcing_term_loc[d];
            for (int k=0; k<dofs_per_cell; k++) {
                grad_phi_u[k] = fe_values[velocities].gradient (k, q);
                div_phi_u[k] = fe_values[velocities].divergence (k, q);
                phi_p[k] = fe_values[pressure].value (k, q);
                phi_u[k] = fe_values[velocities].value (k, q);
            }
            for (int i = 0; i < dofs_per_cell; i++) {
                for (int j = 0; j < dofs_per_cell; j++) {
                    cell_matrix(i,j) +=
                    (nu*scalar_product(grad_phi_u[i],grad_phi_u[j])
                    + phi_u[j]
                    *transpose(
                    previous_newton_velocity_gradients[q])
                    *phi_u[i]
                    + previous_newton_velocity_values[q]
                    *transpose(grad_phi_u[j])*phi_u[i]
                    - phi_p[j]*div_phi_u[i]
                    - phi_p[i]*div_phi_u[j])
                    *fe_values.JxW(q);
                }
                int equation_i = fe->system_to_component_index(i).first;
                cell_rhs(i) += (scalar_product(forcing_term_tensor, fe_values[velocities].value(i, q)) + previous_newton_velocity_values[q] *
                                            transpose(previous_newton_velocity_gradients[q])*phi_u[i]) *fe_values.JxW(q);
            }
        }
        cell->get_dof_indices(local_dof_indices);
        //constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
          system_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);
    }
}



void NavierStokesSolver::run_newton_loop(int cycle) {
    int MAX_ITER = 10;
    double TOL = 1e-8;
    int iter = 0;
    double residual = 0;
    //solve Stokes equations for initial guess
    assemble_stokes_system();
    solve();
    previous_newton_step = solution;
    while (iter == 0 || (residual > TOL && iter < MAX_ITER)) {
        assemble_system();
        solve();
        TrilinosWrappers::MPI::BlockVector res_vec = solution;
        res_vec -= previous_newton_step;
        residual = res_vec.l2_norm()/(dof_handler.n_dofs());
        previous_newton_step = solution;
        iter++;
        
        pcout << "Residual = " << std::to_string(residual) << std::endl;
    }
    if (iter == MAX_ITER) {
        pcout << "WARNING: Newton’s method failed to converge\n" << std::endl;
    }
}



void
 NavierStokesSolver::assemble_stokes_system()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();
  const unsigned int n_q_face      = quadrature_face->size();

  FEValues<dim>     fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values(*fe,
                                   *quadrature_face,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix = 0.0;
  system_rhs    = 0.0;
  pressure_mass = 0.0;

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix               = 0.0;
      cell_rhs                  = 0.0;
      cell_pressure_mass_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          Vector<double> forcing_term_loc(dim);
          forcing_term.vector_value(fe_values.quadrature_point(q),
                                    forcing_term_loc);
          Tensor<1, dim> forcing_term_tensor;
          for (unsigned int d = 0; d < dim; ++d)
            forcing_term_tensor[d] = forcing_term_loc[d];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Viscosity term.
                  cell_matrix(i, j) +=
                    nu *
                    scalar_product(fe_values[velocity].gradient(i, q),
                                   fe_values[velocity].gradient(j, q)) *
                    fe_values.JxW(q);

                  // Pressure term in the momentum equation.
                  cell_matrix(i, j) -= fe_values[velocity].divergence(i, q) *
                                       fe_values[pressure].value(j, q) *
                                       fe_values.JxW(q);

                  // Pressure term in the continuity equation.
                  cell_matrix(i, j) -= fe_values[velocity].divergence(j, q) *
                                       fe_values[pressure].value(i, q) *
                                       fe_values.JxW(q);

                  // Pressure mass matrix.
                  cell_pressure_mass_matrix(i, j) +=
                    fe_values[pressure].value(i, q) *
                    fe_values[pressure].value(j, q) / nu * fe_values.JxW(q);
                }

              // Forcing term.
              cell_rhs(i) += scalar_product(forcing_term_tensor,
                                            fe_values[velocity].value(i, q)) *
                             fe_values.JxW(q);
            }
        }

      // Boundary integral for Neumann BCs.
      if (cell->at_boundary())
        {
          for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
              if (cell->face(f)->at_boundary() &&
                  cell->face(f)->boundary_id() == 2)
                {
                  fe_face_values.reinit(cell, f);

                  for (unsigned int q = 0; q < n_q_face; ++q)
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          cell_rhs(i) +=
                            -p_out *
                            scalar_product(fe_face_values.normal_vector(q),
                                           fe_face_values[velocity].value(i,
                                                                          q)) *
                            fe_face_values.JxW(q);
                        }
                    }
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
      pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
  pressure_mass.compress(VectorOperation::add);

  // Dirichlet boundary conditions.
  {
    std::map<types::global_dof_index, double>           boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    // We interpolate first the inlet velocity condition alone, then the wall
    // condition alone, so that the latter "win" over the former where the two
    // boundaries touch.
    boundary_functions[0] = &inlet_velocity;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             ComponentMask(
                                               {true, true, true, false}));

    boundary_functions.clear();
    Functions::ZeroFunction<dim> zero_function(dim + 1);
    boundary_functions[1] = &zero_function;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             ComponentMask(
                                               {true, true, true, false}));

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, false);
  }
}

void
 NavierStokesSolver::solve()
{
  pcout << "===============================================" << std::endl;

  SolverControl solver_control(2000, 1e-6 * system_rhs.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

  // PreconditionBlockDiagonal preconditioner;
  // preconditioner.initialize(system_matrix.block(0, 0),
  //                           pressure_mass.block(1, 1));

  PreconditionBlockTriangular preconditioner;
  preconditioner.initialize(system_matrix.block(0, 0),
                            pressure_mass.block(1, 1),
                            system_matrix.block(1, 0));

  pcout << "Solving the linear system" << std::endl;
  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " GMRES iterations"
        << std::endl;

  solution = solution_owned;
}
