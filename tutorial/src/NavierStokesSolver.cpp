#include "NavierStokesSolver.hpp"

 
void StationaryNavierStokes::setup_dofs() {
  system_matrix.clear();
  pressure_mass_matrix.clear();

  dof_handler.distribute_dofs(fe);

  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(dof_handler, block_component);

  dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
  unsigned int dof_u = dofs_per_block[0];
  unsigned int dof_p = dofs_per_block[1];

  const FEValuesExtractors::Vector velocities(0);
  {
    nonzero_constraints.clear();

    DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                              0,
                                              BoundaryValues(),
                                              nonzero_constraints,
                                              fe.component_mask(velocities));
  }
  nonzero_constraints.close();

  {
    zero_constraints.clear();

    DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                              0,
                                              Functions::ZeroFunction<dim>(
                                                dim + 1),
                                              zero_constraints,
                                              fe.component_mask(velocities));
  }
  zero_constraints.close();

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << " (" << dof_u << " + " << dof_p << ')' << std::endl;
}

 
void StationaryNavierStokes::initialize_system()
{
  {
    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
    sparsity_pattern.copy_from(dsp);
  }

  system_matrix.reinit(sparsity_pattern);

  present_solution.reinit(dofs_per_block);
  delta_owned.reinit(dofs_per_block);
  residual_vector.reinit(dofs_per_block);
}

 
void StationaryNavierStokes::assemble(const bool initial_step,
                                            const bool assemble_matrix) {
  if (assemble_matrix)
    system_matrix = 0;

  residual_vector = 0;

  QGauss<dim> quadrature_formula(degree + 2);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values | update_gradients);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     local_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


  std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
  std::vector<double>         present_pressure_values(n_q_points);

  std::vector<double>         div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double>         phi_p(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      local_matrix = 0;
      local_rhs    = 0;

      fe_values[velocities].get_function_values(evaluation_point,
                                                present_velocity_values);

      fe_values[velocities].get_function_gradients(
        evaluation_point, present_velocity_gradients);

      fe_values[pressure].get_function_values(evaluation_point,
                                              present_pressure_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              div_phi_u[k]  = fe_values[velocities].divergence(k, q);
              grad_phi_u[k] = fe_values[velocities].gradient(k, q);
              phi_u[k]      = fe_values[velocities].value(k, q);
              phi_p[k]      = fe_values[pressure].value(k, q);
            }

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              if (assemble_matrix)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      local_matrix(i, j) +=
                        (viscosity * scalar_product(grad_phi_u[j], grad_phi_u[i]) +
                          present_velocity_gradients[q] * phi_u[j] * phi_u[i] +
                          grad_phi_u[j] * present_velocity_values[q] *
                            phi_u[i] -
                          div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j] +
                          gamma * div_phi_u[j] * div_phi_u[i] +
                          phi_p[i] * phi_p[j]) *
                        fe_values.JxW(q);
                    }
                }

              double present_velocity_divergence = trace(present_velocity_gradients[q]);
              local_rhs(i) +=
                (-viscosity * scalar_product(present_velocity_gradients[q],
                                              grad_phi_u[i]) -
                  present_velocity_gradients[q] * present_velocity_values[q] *
                    phi_u[i] +
                  present_pressure_values[q] * div_phi_u[i] +
                  present_velocity_divergence * phi_p[i] -
                  gamma * present_velocity_divergence * div_phi_u[i]) *
                fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(local_dof_indices);

      const AffineConstraints<double> &constraints_used = initial_step ? nonzero_constraints : zero_constraints;

      if (assemble_matrix) {
          constraints_used.distribute_local_to_global(local_matrix,
                                                      local_rhs,
                                                      local_dof_indices,
                                                      system_matrix,
                                                      residual_vector);
        }
      else {
          constraints_used.distribute_local_to_global(local_rhs,
                                                      local_dof_indices,
                                                      residual_vector);
        }
    }

  if (assemble_matrix) {
      pressure_mass_matrix.reinit(sparsity_pattern.block(1, 1));
      pressure_mass_matrix.copy_from(system_matrix.block(1, 1));

      system_matrix.block(1, 1) = 0;
    }
}

 
void StationaryNavierStokes::assemble_system(const bool initial_step) {
  assemble(initial_step, true);
}

 
void StationaryNavierStokes::assemble_rhs(const bool initial_step) {
  assemble(initial_step, false);
}

 
void StationaryNavierStokes::solve(const bool initial_step)
{
  const AffineConstraints<double> &constraints_used =
    initial_step ? nonzero_constraints : zero_constraints;

  SolverControl solver_control(system_matrix.m(),
                                1e-4 * residual_vector.l2_norm(),
                                true);

  SolverFGMRES<BlockVector<double>> gmres(solver_control);
  SparseILU<double>                 pmass_preconditioner;
  pmass_preconditioner.initialize(pressure_mass_matrix,
                                  SparseILU<double>::AdditionalData());

  const BlockSchurPreconditioner<SparseILU<double>> preconditioner(
    gamma,
    viscosity,
    system_matrix,
    pressure_mass_matrix,
    pmass_preconditioner);

  gmres.solve(system_matrix, delta_owned, residual_vector, preconditioner);
  std::cout << "FGMRES steps: " << solver_control.last_step() << std::endl;

  constraints_used.distribute(delta_owned);
}


 
void StationaryNavierStokes::newton_iteration(
  const double       tolerance,
  const bool         is_initial_step,
  const bool         output_result)
{
  bool first_step = is_initial_step;

    double       last_res      = 1.0;
    double       current_res   = 1.0;
    std::cout << "viscosity: " << viscosity << std::endl;

    while ((first_step || current_res > tolerance))
      {
        if (first_step)
          {
            setup_dofs();
            initialize_system();
            evaluation_point = present_solution;
            assemble_system(first_step);
            solve(first_step);
            present_solution = delta_owned;
            nonzero_constraints.distribute(present_solution);
            first_step       = false;
            evaluation_point = present_solution;
            assemble_rhs(first_step);
            current_res = residual_vector.l2_norm();
            std::cout << "The residual of initial guess is " << current_res
                      << std::endl;
            last_res = current_res;
          }
        else
          {
            evaluation_point = present_solution;
            assemble_system(first_step);
            solve(first_step);

            for (double alpha = 1.0; alpha > 1e-5; alpha *= 0.5)
              {
                evaluation_point = present_solution;
                evaluation_point.add(alpha, delta_owned);
                nonzero_constraints.distribute(evaluation_point);
                assemble_rhs(first_step);
                current_res = residual_vector.l2_norm();
                std::cout << "  alpha: " << std::setw(10) << alpha
                          << std::setw(0) << "  residual: " << current_res
                          << std::endl;
                if (current_res < last_res)
                  break;
              }
            {
              present_solution = evaluation_point;
              std::cout << "  residual: " << current_res << std::endl;
              last_res = current_res;
            }
          }

        if (output_result) output_results(69);
          
      }

  
}

 
void StationaryNavierStokes::compute_initial_guess()
{ 
  const double re_increment = 2000.0;
  const double max_RE = 1.0 / viscosity;

  bool is_initial_step = true;

  for (double Re = 1000.0; Re < max_RE; Re = std::min(Re + re_increment, max_RE))
    {
      viscosity = 1.0 / Re;
      std::cout << "Searching for initial guess with Re = " << Re << std::endl;
      newton_iteration(1e-12, is_initial_step, false);
      is_initial_step = false;
    }
}



 
void StationaryNavierStokes::run(const unsigned int refinement)
{
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(5);

  const double Re = 1.0 / viscosity;

  if (Re > 1000.0)
    {
      std::cout << "Searching for initial guess ..." << std::endl;
      compute_initial_guess();
      std::cout << "Found initial guess." << std::endl;
      std::cout << "Computing solution with Re = " << Re << std::endl;
      viscosity = 1.0 / Re;
      newton_iteration(1e-12, false, true);
    }
  else
    {
      std::cout << "Reynolds is now: " << Re << " *******************************************" << std::endl;
      newton_iteration(1e-12,  true, true);
    }
}


 
void StationaryNavierStokes::output_results(const unsigned int output_index) const {
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.emplace_back("pressure");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(present_solution,
                            solution_names,
                            DataOut<dim>::type_dof_data,
                            data_component_interpretation);
  data_out.build_patches();

  std::ofstream output(std::to_string(1.0 / viscosity) + "-solution-" +
                        Utilities::int_to_string(output_index, 4) + ".vtk");
  data_out.write_vtk(output);
}