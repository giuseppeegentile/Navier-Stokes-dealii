
#include "NavierStokesSolver.hpp"


void
NavierStokesSolver::setup()
{
  // Create the mesh.
  {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh);

    std::ifstream grid_in_file("../mesh/mesh-problem-" +
                               std::to_string(subdomain_id) + ".msh");

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
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);
  }

  // Initialize the linear system.
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
  }
}


void NavierStokesSolver::assemble() {

      pcout << "===============================================" << std::endl;
    pcout << "Assembling the system" << std::endl;

    //clear the system matrix and rhs, because these change every iteration
    system_matrix.reinit(sparsity_pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    QGauss<dim> quadrature_formula (fe->degree + 1);
    const int dofs_per_cell = fe->dofs_per_cell;
    const int n_q_points = quadrature_formula.size();
    const int n_q_points_face = face_quadrature_formula.size();
    std::vector<Tensor<1,dim> > previous_newton_velocity_values (n_q_points);
    std::vector<Tensor< 2, dim> > previous_newton_velocity_gradients (n_q_points);
    std::vector<Vector<double> > rhs_values (n_q_points, Vector<double>(dim+1));
    std::vector<Tensor<2,dim> > grad_phi_u(dofs_per_cell);
    std::vector<double> div_phi_u(dofs_per_cell);
    std::vector<double> phi_p(dofs_per_cell);
    std::vector<Tensor<1,dim> > phi_u(dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    FEValues<dim> fe_values(*fe, *quadrature_formula, update_values |
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
        forcing_function->vector_value_list(fe_values.get_quadrature_points(), rhs_values);
        //calculate cell contribution to system
        for (int q = 0; q < n_q_points; q++) {
            for (int k=0; k<dofs_per_cell; k++) {
                grad_phi_u[k] = fe_values[velocities].gradient (k, q);
                div_phi_u[k] = fe_values[velocities].divergence (k, q);
                phi_p[k] = fe_values[pressure].value (k, q);
                phi_u[k] = fe_values[velocities].value (k, q);
                }
            for (int i = 0; i < dofs_per_cell; i++) {
                for (int j = 0; j < dofs_per_cell; j++) {
                    cell_matrix(i,j) +=
                    (input.nu*double_contract(grad_phi_u[i],grad_phi_u[j])
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
                int equation_i = fe.system_to_component_index(i).first;
                cell_rhs(i) += (fe_values.shape_value(i,q)*rhs_values[q](equation_i) + previous_newton_velocity_values[q] *
                                            transpose(previous_newton_velocity_gradients[q])*phi_u[i]) *fe_values.JxW(q);
            }
        }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
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
        assemble_system(input.nu);
        solve();
        Vector<double> res_vec = solution;
        res_vec -= previous_newton_step;
        residual = res_vec.l2_norm()/(dof_handler.n_dofs());
        previous_newton_step = solution;
        iter++;
        
        pcout << "Residual = ", residual << std::endl;;
    }
    if (iter == MAX_ITER) {
        pcout << "WARNING: Newton’s method failed to converge\n" << std::endl;
    }
}