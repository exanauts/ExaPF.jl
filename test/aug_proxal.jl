@testset "AugLagEvaluators" begin
    datafile = joinpath(dirname(@__FILE__), "..", "data", "case57.m")
    # Build reference evaluator
    nlp = ExaPF.ReducedSpaceEvaluator(datafile)
    time = ExaPF.Normal
    proxal = ExaPF.ProxALEvaluator(nlp, time)
    w0 = ExaPF.initial(proxal)
    w♭, w♯ = ExaPF.bounds(proxal, ExaPF.Variables())
    # Build penalty evaluator
    for scaling in [false, true]
        pen = ExaPF.AugLagEvaluator(proxal, w0; scale=scaling)
        w = w♭
        # Update nlp to stay on manifold
        ExaPF.update!(pen, w)
        # Compute objective
        c = ExaPF.objective(pen, w)
        c_ref = ExaPF.inner_objective(pen, w)
        @test isa(c, Real)
        # For case57.m some constraints are active, so penalty are >= 0
        @test c > c_ref
        inf_pr2 = ExaPF.primal_infeasibility(pen, w)
        @test inf_pr2 > 0.0

        # Update penalty weigth with a large factor to have
        # a meaningful derivative check
        ExaPF.update_penalty!(pen, η=1e3)
        ExaPF.update_multipliers!(pen)
        ExaPF.update!(pen, w)
        obj = ExaPF.objective(pen, w)
        # Compute gradient of objective
        g = similar(w)
        fill!(g, 0)
        ExaPF.gradient!(pen, g, w)
        # Compare with finite differences
        function reduced_cost(w_)
            ExaPF.update!(pen, w_)
            return ExaPF.objective(pen, w_)
        end
        grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, w)
        @test isapprox(grad_fd, g, rtol=1e-6)
    end
end

