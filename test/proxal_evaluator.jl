using FiniteDiff
using Test

@testset "ProxALEvaluators ($time)" for time in [ExaPF.Origin, ExaPF.Normal, ExaPF.Final]
    datafile = joinpath(dirname(@__FILE__), "..", "data", "case9.m")
    # Build reference evaluator
    nlp = ExaPF.ReducedSpaceEvaluator(datafile)
    S = ExaPF.type_array(nlp)

    u0 = ExaPF.initial(nlp)
    # Build ProxAL evaluator
    prox = ExaPF.ProxALEvaluator(nlp, time)

    n = ExaPF.n_variables(prox)
    w = ExaPF.initial(prox)
    @test length(w) == n

    # Update nlp to stay on manifold
    conv = ExaPF.update!(prox, w)
    @test conv.has_converged

    # Compute objective
    c = ExaPF.objective(prox, w)

    @testset "Gradient" begin
        g = similar(w) ; fill!(g, 0)
        ExaPF.gradient!(prox, g, w)

        # Test evaluation of gradient with Finite Differences
        function reduced_cost(w_)
            ExaPF.update!(prox, w_)
            return ExaPF.objective(prox, w_)
        end
        grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, w)
        @test isapprox(grad_fd, g, rtol=1e-6)

        # Test gradient with non-trivial penalties
        λf = 0.5 * rand(prox.ng)
        λt = 1.5 * rand(prox.ng)
        pgf = rand(prox.ng)
        ExaPF.update_primal!(prox, ExaPF.Previous(), pgf)
        ExaPF.update_multipliers!(prox, ExaPF.Next(), λt)
        ExaPF.update_multipliers!(prox, ExaPF.Current(), λf)

        ExaPF.update!(prox, w)
        fill!(g, 0)
        ExaPF.gradient!(prox, g, w)
        grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, w)
        @test isapprox(grad_fd, g, rtol=1e-6)
    end

    @testset "Constraints" begin
        m_I = ExaPF.n_constraints(prox)
        cons = ExaPF.xzeros(S, m_I)
        # Evaluate constraints
        ExaPF.constraint!(prox, cons, w)
        # Transpose Jacobian vector product
        v = ExaPF.xzeros(S, m_I)
        jv = ExaPF.xzeros(S, n)
        ExaPF.jtprod!(prox, jv, w, v)

        # Jacobian structure
        rows, cols = ExaPF.jacobian_structure(prox)
        # Evaluation
        M = Array
        jac = M{Float64, 2}(undef, m_I, n)
        fill!(jac, 0)
        ExaPF.jacobian!(prox, jac, w)
        # Check correctness of transpose Jacobian vector product
        @test jv == jac' * v
    end

    ExaPF.reset!(prox)
end

