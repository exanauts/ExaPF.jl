using FiniteDiff
using Test
import ExaPF: PS

@testset "ProxALEvaluators ($time)" for time in [ExaPF.Origin, ExaPF.Normal, ExaPF.Final]
    datafile = joinpath(INSTANCES_DIR, "case9.m")

    @testset "PowerNetwork Constructor" begin
        pf = PS.PowerNetwork(datafile)
        eps = 1e-10
        powerflow_solver = NewtonRaphson(tol=eps)
        prox0 = ExaPF.ProxALEvaluator(pf, time; powerflow_solver=powerflow_solver)
        @test isa(prox0, ExaPF.ProxALEvaluator)
        @test isa(prox0.inner, ExaPF.ReducedSpaceEvaluator)
        # Test that argument was correctly set
        @test prox0.inner.powerflow_solver == powerflow_solver
    end

    # Build reference evaluator
    nlp = ExaPF.ReducedSpaceEvaluator(datafile; powerflow_solver=NewtonRaphson(tol=1e-12))
    S = ExaPF.array_type(nlp)

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

    @testset "Gradient & Hessian" begin
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

        hv = similar(w) ; fill!(hv, 0)
        tgt = similar(w) ; fill!(tgt, 0)
        tgt[1] = 1.0
        ExaPF.hessprod!(prox, hv, w, tgt)
        H = ExaPF.hessian(prox, w)

        hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, w)
        # Take attribute data as hess_fd is of type Symmetric
        @test H ≈ hess_fd.data rtol=1e-6
    end

    @testset "Constraints" begin
        m_I = ExaPF.n_constraints(prox)
        cons = similar(w, m_I) ; fill!(cons, 0)
        # Evaluate constraints
        ExaPF.constraint!(prox, cons, w)
        # Transpose Jacobian vector product
        v = similar(w, m_I) ; fill!(v, 0)
        jv = similar(w, n) ; fill!(jv, 0)
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

