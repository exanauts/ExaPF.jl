@testset "ReducedSpaceEvaluators ($case)" for case in ["case9.m", "case30.m"]
    if has_cuda_gpu()
        ITERATORS = zip([CPU(), CUDADevice()], [Array, CuArray])
    else
        ITERATORS = zip([CPU()], [Array])
    end
    datafile = joinpath(dirname(@__FILE__), "..", "data", case)
    pf = PowerSystem.PowerNetwork(datafile, 1)

    @testset "Test API on $device" for (device, M) in ITERATORS
        polar = PolarForm(pf, device)
        x0 = ExaPF.initial(polar, State())
        u0 = ExaPF.initial(polar, Control())
        p = ExaPF.initial(polar, Parameters())

        constraints = Function[ExaPF.state_constraint, ExaPF.power_constraints]
        nlp = ExaPF.ReducedSpaceEvaluator(polar, x0, u0, p; constraints=constraints)
        # Test printing
        println(nlp)

        # Test evaluator is well instantiated on target device
        TNLP = typeof(nlp)
        for (fn, ft) in zip(fieldnames(TNLP), fieldtypes(TNLP))
            if ft <: AbstractArray{Float64, P} where P
                @test isa(getfield(nlp, fn), M)
            end
        end

        # Test consistence
        n = ExaPF.n_variables(nlp)
        m = ExaPF.n_constraints(nlp)
        @test n == length(u0)
        @test isless(nlp.u_min, nlp.u_max)
        @test isless(nlp.x_min, nlp.x_max)
        @test isless(nlp.g_min, nlp.g_max)
        @test length(nlp.g_min) == m

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
        @test isapprox(grad_fd, g, rtol=1e-4)

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
        @test nlp.x == x0
        @test iszero(nlp.λ)
    end
end

