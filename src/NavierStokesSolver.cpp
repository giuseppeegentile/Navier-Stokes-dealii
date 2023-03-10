#include "NavierStokesSolver.hpp"

void
NavierStokesSolver::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file("../mesh/correct_mesh_yt.msh");
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
    const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);
    fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity,
                                         dim,
                                         fe_scalar_pressure,
                                         1);

    pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree
          << std::endl;
    pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree
          << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;

    quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

    pcout << "  Quadrature points per face = " << quadrature_face->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We want to reorder DoFs so that all velocity DoFs come first, and then
    // all pressure DoFs.
    std::vector<unsigned int> block_component(dim + 1, 0); /* Vector with 4 elements: 3 components of velocity and 1 of pressure.
                                                              The first 3 elements are 0 and the last one... */
    block_component[dim] = 1; /* ...is 1.
                                 Meaning that: I want the first 3 components to belong to block 0 and the last one to block 1. */
    DoFRenumbering::component_wise(dof_handler, block_component); /* Indexes are changed in the way I want. */

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // Besides the locally owned and locally relevant indices for the whole
    // system (velocity and pressure), we will also need those for the
    // individual velocity and pressure blocks.
    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    block_owned_dofs.resize(2);
    block_relevant_dofs.resize(2);
    block_owned_dofs[0]    = locally_owned_dofs.get_view(0, n_u);
    block_owned_dofs[1]    = locally_owned_dofs.get_view(n_u, n_u + n_p);
    block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
    block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    pcout << "  Number of DoFs: " << std::endl;
    pcout << "    velocity = " << n_u << std::endl;
    pcout << "    pressure = " << n_p << std::endl;
    pcout << "    total    = " << n_u + n_p << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "  Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs, /* !!! It's not the single IndexSet. */
                                                    MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    /* The DoF handler sees all the DoFs as part of the same system; it does not really know that 
       velocity interacts with velocity, velocity with pressure and not pressure with itself. */

    // Velocity DoFs interact with other velocity DoFs (the weak formulation has
    // terms involving u times v), and pressure DoFs interact with velocity DoFs
    // (there are terms involving p times v or u times q). However, pressure
    // DoFs do not interact with other pressure DoFs (there are no terms
    // involving p times q). We build a table to store this information, so that
    // the sparsity pattern can be built accordingly. /* (Saving memory) */

    /* I want to tell deal.II: "Insert slots in my matrix to couple velocity with itself, velocity and pressure,
                                but I don't need slots to couple pressure with itself". */
    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1); /* This information is stored in this coupling table 
                                                                (we can think of it as if it was a matrix once again). */
    for (unsigned int c = 0; c < dim + 1; ++c)
      {
        for (unsigned int d = 0; d < dim + 1; ++d)
          {
            if (c == dim && d == dim) // pressure-pressure term
              coupling[c][d] = DoFTools::none; /* I don't want you to insert entries for this term. */
            else // other combinations
              coupling[c][d] = DoFTools::always; /* I do want you to insert these terms. */
          }
      }

    TrilinosWrappers::BlockSparsityPattern sparsity_stokes(block_owned_dofs, /* !!! It's not the single IndexSet. */
                                                    MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity_stokes);
    sparsity_stokes.compress();

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

    pcout << "  Initializing the matrices" << std::endl;
    jacobian_matrix.reinit(sparsity);
    stokes_system_matrix.reinit(sparsity_stokes);
    pressure_mass.reinit(sparsity_pressure_mass);
    stokes_pressure_mass.reinit(sparsity_pressure_mass);

    pcout << "  Initializing the system right-hand side" << std::endl;
    residual_vector.reinit(block_owned_dofs, MPI_COMM_WORLD);
    stokes_system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
    delta_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);

    solution_old = solution;
  }
}

void
NavierStokesSolver::assemble_system()
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
  Vector<double>     cell_residual(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian_matrix = 0.0;
  residual_vector = 0.0;
  pressure_mass = 0.0;

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  // We use these vectors to store the old solution (i.e. at previous Newton
  // iteration) and its gradient on quadrature nodes of the current cell.
  std::vector<Tensor<1, dim>> present_velocity_values(n_q);
  std::vector<Tensor<2, dim>> present_velocity_gradients(n_q);
  std::vector<double>         present_pressure_values(n_q);

  std::vector<Tensor<1, dim>> old_velocity_values(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix               = 0.0;
      cell_residual             = 0.0;
      cell_pressure_mass_matrix = 0.0;

      fe_values[velocity].get_function_values(solution, present_velocity_values);
      fe_values[velocity].get_function_gradients(solution, present_velocity_gradients);
      fe_values[pressure].get_function_values(solution, present_pressure_values);

      fe_values[velocity].get_function_values(solution_old, old_velocity_values);

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
                  // Mass matrix.
                  cell_matrix(i, j) += fe_values[velocity].value(i, q) *
                                       fe_values[velocity].value(j, q) / deltat *
                                       fe_values.JxW(q);

                  // Viscosity term.
                  cell_matrix(i, j) += nu * rho *
                                       scalar_product(fe_values[velocity].gradient(i, q),
                                                      fe_values[velocity].gradient(j, q)) *
                                       fe_values.JxW(q);

                  cell_matrix(i, j) += rho *
                                       present_velocity_gradients[q] *
                                       fe_values[velocity].value(j, q) *
                                       fe_values[velocity].value(i, q) *
                                       fe_values.JxW(q);

                  cell_matrix(i, j) += rho *
                                       present_velocity_values[q] *
                                       fe_values[velocity].gradient(j, q) *
                                       fe_values[velocity].value(i, q) *
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
                  cell_pressure_mass_matrix(i, j) += fe_values[pressure].value(i, q) *
                                                     fe_values[pressure].value(j, q) / nu * 
                                                     fe_values.JxW(q);
                }

              // Time derivative term.
              cell_residual(i) -= rho * (present_velocity_values[q] - old_velocity_values[q]) / deltat * 
                                  fe_values[velocity].value(i, q) *
                                  fe_values.JxW(q);

              cell_residual(i) -= nu * rho *
                                  scalar_product(present_velocity_gradients[q],
                                                 fe_values[velocity].gradient(i, q)) *
                                  fe_values.JxW(q);

              cell_residual(i) -= rho *
                                  present_velocity_values[q] *
                                  present_velocity_gradients[q] *
                                  fe_values[velocity].value(i, q) *
                                  fe_values.JxW(q);

              cell_residual(i) += present_pressure_values[q] *
                                  fe_values[velocity].divergence(i, q) *
                                  fe_values.JxW(q);

              // Forcing term.
              cell_residual(i) += scalar_product(forcing_term_tensor,
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
                  cell->face(f)->boundary_id() == 10)
                {
                  fe_face_values.reinit(cell, f);

                  for (unsigned int q = 0; q < n_q_face; ++q)
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          cell_residual(i) += -p_out *
                                              scalar_product(fe_face_values.normal_vector(q),
                                                             fe_face_values[velocity].value(i, q)) *
                                              fe_face_values.JxW(q);
                        }
                    }
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      jacobian_matrix.add(dof_indices, cell_matrix);
      residual_vector.add(dof_indices, cell_residual);
      pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
    }

  jacobian_matrix.compress(VectorOperation::add);
  residual_vector.compress(VectorOperation::add);
  pressure_mass.compress(VectorOperation::add);

  // Dirichlet boundary conditions.
  {
    std::map<types::global_dof_index, double>           boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    // We interpolate first the inlet velocity condition alone, then the wall
    // condition alone, so that the latter "win" over the former where the two
    // boundaries touch.
    boundary_functions[11] = &inlet_velocity;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             ComponentMask(
                                               {true, true, false})); /* They're only applied to the velocity */

/*     boundary_functions.clear(); */ /* The order is important because... what about the DoFs on the interface between inlet and wall?
                                   In this case we want wall bcs to win over the inlet, so we write it later. */
    Functions::ZeroFunction<dim> zero_function(dim + 1);
    boundary_functions[12] = &zero_function;
    boundary_functions[13] = &zero_function;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             ComponentMask(
                                               {true, true, false}));

    MatrixTools::apply_boundary_values(
      boundary_values, jacobian_matrix, delta_owned, residual_vector, false);
  }
}

void
NavierStokesSolver::assemble_stokes_system()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the Stokes system" << std::endl;

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

  stokes_system_matrix = 0.0;
  stokes_system_rhs    = 0.0;
  stokes_pressure_mass = 0.0;

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
                    nu * rho *
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
                  cell->face(f)->boundary_id() == 1)
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

      stokes_system_matrix.add(dof_indices, cell_matrix);
      stokes_system_rhs.add(dof_indices, cell_rhs);
      stokes_pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
    }

  stokes_system_matrix.compress(VectorOperation::add);
  stokes_system_rhs.compress(VectorOperation::add);
  stokes_pressure_mass.compress(VectorOperation::add);

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
                                               {true, true, false}));

    boundary_functions.clear();
    Functions::ZeroFunction<dim> zero_function(dim + 1);
    boundary_functions[2] = &zero_function;
    boundary_functions[3] = &zero_function;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             ComponentMask(
                                               {true, true, false}));

    MatrixTools::apply_boundary_values(
      boundary_values, stokes_system_matrix, solution, stokes_system_rhs, false);
  }
}

void
NavierStokesSolver::solve_stokes_system()
{
  pcout << "===============================================" << std::endl;

  SolverControl solver_control(2000, 1e-6 * stokes_system_rhs.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

  // PreconditionBlockDiagonal preconditioner;
  // preconditioner.initialize(system_matrix.block(0, 0),
  //                           pressure_mass.block(1, 1));

  PreconditionBlockTriangular preconditioner;
  preconditioner.initialize(stokes_system_matrix.block(0, 0),
                            stokes_pressure_mass.block(1, 1),
                            stokes_system_matrix.block(1, 0));

  pcout << "Solving the Stokes system" << std::endl;
  solver.solve(stokes_system_matrix, solution_owned, stokes_system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " GMRES iterations"
        << std::endl;

  solution = solution_owned;

  output(0., 0.);
}

void
NavierStokesSolver::solve_system()
{
  pcout << "===============================================" << std::endl;

  SolverControl solver_control(100000, 1e-2 * residual_vector.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

  PreconditionIdentity preconditioner;

  // PreconditionBlockDiagonal preconditioner;
  // preconditioner.initialize(jacobian_matrix.block(0, 0),
  //                           pressure_mass.block(1, 1));


  // PreconditionBlockTriangular preconditioner;
  // preconditioner.initialize(jacobian_matrix.block(0, 0),
  //                           pressure_mass.block(1, 1),
  //                           jacobian_matrix.block(1, 0));

  pcout << "Solving system..." << std::endl;
  solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner); 
  pcout << "   " << solver_control.last_step() << " GMRES iterations"
        << std::endl;

  solution = solution_owned;
}

void
NavierStokesSolver::solve_newton()
{
  const unsigned int n_max_iters        = 1000;
  const double       residual_tolerance = 1e-2;

  unsigned int n_iter        = 0;
  double       residual_norm = residual_tolerance + 1;

  while (n_iter < n_max_iters && residual_norm > residual_tolerance)
    {
      assemble_system();
      residual_norm = residual_vector.l2_norm();

      pcout << "  Newton iteration " << n_iter << "/" << n_max_iters
            << " - ||r|| = " << std::scientific << std::setprecision(6)
            << residual_norm << std::flush;

      // We actually solve the system only if the residual is larger than the
      // tolerance.
      if (residual_norm > residual_tolerance)
        {
          solve_system();
          pcout << "System solved!" << std::endl;

          // delta_owned *= 0.1;
          solution_owned += delta_owned;
          // solution_owned.add(0.001, delta_owned);
          solution = solution_owned;
        }
      else
        {
          pcout << " < tolerance" << std::endl;
        }

      ++n_iter;
    }
}

void
NavierStokesSolver::solve()
{
  pcout << "===============================================" << std::endl;

  time = 0.0;

/*   // Finding the initial condition (small Reynolds number).
  {
    pcout << "Finding the initial condition" << std::endl;

    assemble_stokes_system();
    solve_stokes_system();

    pcout << "-----------------------------------------------" << std::endl;
  } */

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    output(0, 0.0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T - 0.5 * deltat)
    {
      time += deltat;
      ++time_step;

      // Store the old solution, so that it is available for assembly.
      solution_old = solution;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << std::fixed << time << std::endl;

      // At every time step, we invoke Newton's method to solve the non-linear
      // problem.
      solve_newton();

      output(time_step, time);

      pcout << std::endl;
    }
}

void
NavierStokesSolver::output(const unsigned int &time_step, const double &time) const
{
  pcout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);
  std::vector<std::string> names = {"velocity",
                                    "velocity",
                                    "pressure"};

  data_out.add_data_vector(dof_handler,
                           solution,
                           names,
                           data_component_interpretation);

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  std::string output_file_name = std::to_string(time_step);

  // Pad with zeros.
  output_file_name = "output-" + std::string(4 - output_file_name.size(), '0') +
                     output_file_name;

  DataOutBase::DataOutFilter data_filter(
    DataOutBase::DataOutFilterFlags(/*filter_duplicate_vertices = */ false,
                                    /*xdmf_hdf5_output = */ true));
  data_out.write_filtered_data(data_filter);
  data_out.write_hdf5_parallel(data_filter,
                               output_file_name + ".h5",
                               MPI_COMM_WORLD);

  std::vector<XDMFEntry> xdmf_entries({data_out.create_xdmf_entry(
    data_filter, output_file_name + ".h5", time, MPI_COMM_WORLD)});
  data_out.write_xdmf_file(xdmf_entries,
                           output_file_name + ".xdmf",
                           MPI_COMM_WORLD);
}

/* void NavierStokesSolver::assemble_system() {

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
} */



/* void NavierStokesSolver::run_newton_loop(int cycle) {
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
        pcout << "WARNING: Newton???s method failed to converge\n" << std::endl;
    }
} */