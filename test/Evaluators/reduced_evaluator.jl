@testset "ReducedSpaceEvaluators ($case)" for case in ["case9.m", "case30.m"]
    if has_cuda_gpu()
        ITERATORS = zip([CPU(), CUDADevice()], [Array, CuArray])
    else
        ITERATORS = zip([CPU()], [Array])
    end
    datafile = joinpath(INSTANCES_DIR, case)
    @testset "Constructor" for (device, M) in ITERATORS
        powerflow_solver = NewtonRaphson(; tol=1e-11)
        nlp = ExaPF.ReducedSpaceEvaluator(datafile; device=device, powerflow_solver=powerflow_solver)
        # Test that arguments are passed as expected in the constructor:
        @test nlp.powerflow_solver == powerflow_solver
    end

    pf = PowerSystem.PowerNetwork(datafile)
    @testset "Test API on $device" for (device, M) in ITERATORS
        polar = PolarForm(pf, device)
        x0 = ExaPF.initial(polar, State())
        u0 = ExaPF.initial(polar, Control())

        constraints = Function[
            ExaPF.voltage_magnitude_constraints,
            ExaPF.active_power_constraints,
            ExaPF.reactive_power_constraints,
            ExaPF.flow_constraints,
        ]
        nlp = ExaPF.ReducedSpaceEvaluator(polar; constraints=constraints)
        # Test printing
        println(devnull, nlp)

        # Test consistence
        n_state = get(polar, NumberOfState())
        @test length(nlp.λ) == n_state
        n = ExaPF.n_variables(nlp)
        m = ExaPF.n_constraints(nlp)
        @test n == length(u0)
        @test isless(nlp.u_min, nlp.u_max)
        @test isless(nlp.g_min, nlp.g_max)
        @test length(nlp.g_min) == m

        # Test API
        @test isa(get(nlp, ExaPF.Constraints()), Array{Function})
        @test get(nlp, State()) == x0
        buffer = get(nlp, ExaPF.PhysicalState())
        @test isa(buffer, ExaPF.AbstractBuffer)
        @test ExaPF.initial(nlp) == u0

        vm, va, pg, qg = buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj

        u = u0
        # Update nlp to stay on manifold
        ExaPF.update!(nlp, u)
        # Compute objective
        c = ExaPF.objective(nlp, u)
        @test isa(c, Real)
        # Compute gradient of objective
        g = similar(u)
        fill!(g, 0)
        ExaPF.gradient!(nlp, g, u)
        function reduced_cost(u_)
            ExaPF.update!(nlp, u_)
            return ExaPF.objective(nlp, u_)
        end
        grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, u)
        @test isapprox(grad_fd, g, rtol=1e-6)

        ExaPF.update!(nlp, u)
        hv = similar(u) ; fill!(hv, 0)
        w = similar(u) ; fill!(w, 0)
        w[1] = 1.0
        ExaPF.hessprod!(nlp, hv, u, w)
        H = similar(u, n, n) ; fill!(H, 0)
        ExaPF.hessian!(nlp, H, u)
        # Is Hessian vector product relevant?
        @test H * w == hv
        # Is Hessian correct?
        hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, u)
        # Take attribute data as hess_fd is of type Symmetric
        @test H ≈ hess_fd.data rtol=1e-6

        # Constraint
        ## Evaluation of the constraints
        cons = similar(nlp.g_min)
        fill!(cons, 0)
        ExaPF.constraint!(nlp, cons, u)

        ## Evaluation of the transpose-Jacobian product
        v = similar(cons) ; fill!(v, 0)
        fill!(g, 0)
        ExaPF.jtprod!(nlp, g, u, v)
        @test iszero(g)
        fill!(v, 1) ; fill!(g, 0)
        ExaPF.jtprod!(nlp, g, u, v)

        ## Evaluation of the Jacobian (only on CPU)
        if isa(device, CPU)
            jac = M{Float64, 2}(undef, m, n)
            ExaPF.jacobian!(nlp, jac, u)
            # Test transpose Jacobian vector product
            @test isapprox(g, transpose(jac) * v)
            # Test Jacobian vector product
            S = typeof(u)
            jv = ExaPF.xzeros(S, m)
            v = ExaPF.xzeros(S, n)
            ExaPF.jprod!(nlp, jv, u, v)
            @test jac * v == jv
        end

        # Utils
        inf_pr1 = ExaPF.primal_infeasibility(nlp, u)
        inf_pr2 = ExaPF.primal_infeasibility!(nlp, cons, u)
        @test inf_pr1 == inf_pr2

        # test reset!
        ExaPF.reset!(nlp)
        @test get(nlp, State()) == x0
        @test iszero(nlp.λ)

        # setters
        nbus = get(nlp, PS.NumberOfBuses())
        loads = similar(u, nbus) ; fill!(loads, 1)
        ExaPF.setvalues!(nlp, PS.ActiveLoad(), loads)
        ExaPF.setvalues!(nlp, PS.ReactiveLoad(), loads)
    end
end

