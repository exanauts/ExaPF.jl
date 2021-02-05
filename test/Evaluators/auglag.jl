
function init(datafile, ::Type{ExaPF.ReducedSpaceEvaluator})
    constraints = Function[
        ExaPF.state_constraints,
        ExaPF.power_constraints,
        # ExaPF.flow_constraints,
    ]
    return ExaPF.ReducedSpaceEvaluator(datafile; constraints=constraints)
end
function init(datafile, ::Type{ExaPF.ProxALEvaluator})
    nlp = ExaPF.ReducedSpaceEvaluator(datafile)
    time = ExaPF.Normal
    return ExaPF.ProxALEvaluator(nlp, time)
end
function init(datafile, ::Type{ExaPF.SlackEvaluator})
    return ExaPF.SlackEvaluator(datafile)
end

@testset "AugLagEvaluator with $Evaluator backend" for Evaluator in [
    ExaPF.ReducedSpaceEvaluator,
    ExaPF.ProxALEvaluator,
    ExaPF.SlackEvaluator,
]
    @testset "Inactive constraints" begin
        datafile = joinpath(INSTANCES_DIR, "case9.m")
        # Build reference evaluator
        nlp = init(datafile, Evaluator)
        u0 = ExaPF.initial(nlp)
        # Build penalty evaluator
        pen = ExaPF.AugLagEvaluator(nlp, u0)

        u = u0
        # Update nlp to stay on manifold
        ExaPF.update!(pen, u)
        # Compute objective
        c = ExaPF.objective(pen, u)
        c_ref = ExaPF.inner_objective(pen, u)
        @test isa(c, Real)
        @test c >= c_ref

        # Compute gradient of objective
        g = similar(u)
        g_ref = similar(u)
        fill!(g, 0)
        ExaPF.gradient!(pen, g, u)
        fill!(g_ref, 0)
        function reduced_cost(u_)
            ExaPF.update!(pen, u_)
            return ExaPF.objective(pen, u_)
        end
        g_ref = FiniteDiff.finite_difference_gradient(reduced_cost, u)
        @test isapprox(g_ref, g, rtol=1e-5)
        # Update penalty weigth
        ExaPF.update_penalty!(pen)
        # Utils
        inf_pr1 = ExaPF.primal_infeasibility(nlp, u)
        @test inf_pr1 >= 0.0
        # Test reset
        ExaPF.reset!(pen)
    end
    @testset "Active constraints" begin
        datafile = joinpath(INSTANCES_DIR, "case57.m")
        # Build reference evaluator
        nlp = init(datafile, Evaluator)
        u0 = ExaPF.initial(nlp)
        w♭, w♯ = ExaPF.bounds(nlp, ExaPF.Variables())
        # Build penalty evaluator
        for scaling in [true, false]
            pen = ExaPF.AugLagEvaluator(nlp, u0; scale=scaling)
            u = w♭
            # Update nlp to stay on manifold
            ExaPF.update!(pen, u)
            # Compute objective
            c = ExaPF.objective(pen, u)
            c_ref = ExaPF.inner_objective(pen, u)
            @test isa(c, Real)
            # For case57.m some constraints are active, so penalty are >= 0
            @test c > c_ref
            inf_pr2 = ExaPF.primal_infeasibility(pen, u)
            @test inf_pr2 > 0.0

            pen.ρ = 10.0
            # Update penalty weigth with a large factor to have
            # a meaningful derivative check
            ExaPF.update_penalty!(pen, η=1e3)
            ExaPF.update_multipliers!(pen)
            ExaPF.update!(pen, u)
            obj = ExaPF.objective(pen, u)
            # Compute gradient of objective
            g = similar(u)
            fill!(g, 0)
            ExaPF.gradient!(pen, g, u)
            # Compare with finite differences
            function reduced_cost(u_)
                ExaPF.update!(pen, u_)
                return ExaPF.objective(pen, u_)
            end
            grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, u)
            @test isapprox(grad_fd, g, rtol=1e-6)

            # Test Hessian only on ReducedSpaceEvaluator and SlackEvaluator
            if isa(nlp, ExaPF.ReducedSpaceEvaluator) || isa(nlp, ExaPF.SlackEvaluator)
                n = length(u)
                ExaPF.update!(pen, u)
                hv = similar(u) ; fill!(hv, 0)
                w = similar(u) ; fill!(w, 0)
                w[1] = 1
                ExaPF.hessprod!(pen, hv, u, w)
                H = similar(u, n, n) ; fill!(H, 0)
                ExaPF.hessian!(pen, H, u)
                # Is Hessian vector product relevant?
                @test H * w == hv
                # Is Hessian correct?
                hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, u)
                @test isapprox(H, hess_fd, rtol=1e-6)
            end
            # Test estimation of multipliers only on SlackEvaluator
            if isa(nlp, ExaPF.SlackEvaluator)
                λ = ExaPF.estimate_multipliers(pen, u)
            end
        end
    end
end

