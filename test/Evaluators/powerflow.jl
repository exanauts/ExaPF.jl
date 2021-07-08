function test_powerflow_evaluator(nlp, device, AT)
    # Parameters
    npartitions = 8

    J = ExaPF.jacobian_sparsity(nlp.model, ExaPF.power_balance, State())
    n = size(J, 1)

    precond = LinearSolvers.BlockJacobiPreconditioner(J, npartitions, nlp.model.device)
    # Retrieve initial state of network
    uk = ExaPF.initial(nlp)

    @testset "Powerflow solver $(LinSolver)" for LinSolver in ExaPF.list_solvers(device)
        (LinSolver == LS.DirectSolver) && continue
        algo = LinSolver(J; P=precond)
        nlp.linear_solver = algo
        convergence = ExaPF.update!(nlp, uk)
        @test convergence.has_converged
        @test convergence.norm_residuals < nlp.powerflow_solver.tol
        ExaPF.reset!(nlp)
    end
end
