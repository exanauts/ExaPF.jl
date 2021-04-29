function test_powerflow_evaluator(nlp, device, AT)
    # Parameters
    npartitions = 8

    precond = ExaPF.build_preconditioner(nlp.model; nblocks=npartitions)
    # Retrieve initial state of network
    uk = ExaPF.initial(nlp)

    @testset "Powerflow solver $(LinSolver)" for LinSolver in ExaPF.list_solvers(device)
        (LinSolver == LS.DirectSolver) && continue
        algo = LinSolver(precond)
        nlp.linear_solver = algo
        convergence = ExaPF.update!(nlp, uk)
        @test convergence.has_converged
        @test convergence.norm_residuals < nlp.powerflow_solver.tol
        ExaPF.reset!(nlp)
    end
end
