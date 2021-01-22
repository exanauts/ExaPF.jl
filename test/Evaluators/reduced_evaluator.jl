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
        TNLP = typeof(nlp)
        for (fn, ft) in zip(fieldnames(TNLP), fieldtypes(TNLP))
            if ft <: AbstractArray{Float64, P} where P
                @test isa(getfield(nlp, fn), M)
            end
        end
        # Test that arguments are passed as expected in the constructor:
        @test nlp.powerflow_solver == powerflow_solver
    end

    pf = PowerSystem.PowerNetwork(datafile)
    @testset "Test API on $device" for (device, M) in ITERATORS
        polar = PolarForm(pf, device)
        x0 = ExaPF.initial(polar, State())
        u0 = ExaPF.initial(polar, Control())

        constraints = Function[ExaPF.state_constraint]
        nlp = ExaPF.ReducedSpaceEvaluator(polar, x0, u0; constraints=constraints)
        # Test printing
        println(devnull, nlp)

        # Test evaluator is well instantiated on target device
        TNLP = typeof(nlp)
        for (fn, ft) in zip(fieldnames(TNLP), fieldtypes(TNLP))
            if ft <: AbstractArray{Float64, P} where P
                @test isa(getfield(nlp, fn), M)
            end
        end

        # Test consistence
        n_state = get(polar, NumberOfState())
        @test length(nlp.λ) == n_state
        n = ExaPF.n_variables(nlp)
        m = ExaPF.n_constraints(nlp)
        @test n == length(u0)
        @test isless(nlp.u_min, nlp.u_max)
        @test isless(nlp.x_min, nlp.x_max)
        @test isless(nlp.g_min, nlp.g_max)
        @test length(nlp.g_min) == m

        # Test API
        @test isa(get(nlp, ExaPF.Constraints()), Array{Function})
        @test get(nlp, State()) == x0
        buffer = get(nlp, ExaPF.PhysicalState())
        @test isa(buffer, ExaPF.AbstractBuffer)
        @test isa(get(nlp, ExaPF.AutoDiffBackend()), ExaPF.AutoDiffFactory)
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

        # Hessian
        ExaPF.update!(nlp, u)
        hv = similar(u) ; fill!(hv, 0)
        w = similar(u) ; fill!(w, 0)
        w[1] = 1
        ExaPF.hessprod!(nlp, hv, u, w)
        H = similar(u, n, n) ; fill!(H, 0)
        ExaPF.hessian!(nlp, H, u)
        # Is Hessian vector product relevant?
        @test H * w == hv
        # Is Hessian correct?
        hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, u)
        @test isapprox(H, hess_fd, rtol=1e-6)

        # Constraint
        ## Evaluation of the constraints
        cons = similar(nlp.g_min)
        fill!(cons, 0)
        ExaPF.constraint!(nlp, cons, u)

        ## Evaluation of the Jacobian
        jac = M{Float64, 2}(undef, m, n)
        fill!(jac, 0)
        ExaPF.jacobian!(nlp, jac, u)

        ## Evaluation of the Jacobian transpose product
        v = similar(cons) ; fill!(v, 0)
        fill!(g, 0)
        ExaPF.jtprod!(nlp, g, u, v)
        @test iszero(g)
        fill!(v, 1) ; fill!(g, 0)
        ExaPF.jtprod!(nlp, g, u, v)
        @test isapprox(g, transpose(jac) * v)

        # Utils
        inf_pr1 = ExaPF.primal_infeasibility(nlp, u)
        inf_pr2 = ExaPF.primal_infeasibility!(nlp, v, u)
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

