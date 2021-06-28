function test_auglag_evaluator(nlp, device, MT)
    u0 = ExaPF.initial(nlp)
    w♭, w♯ = ExaPF.bounds(nlp, ExaPF.Variables())
    # Build penalty evaluator
    @testset "Scaling $scaling" for scaling in [true, false]
        ExaPF.reset!(nlp)
        pen = ExaPF.AugLagEvaluator(nlp, u0; scale=scaling)
        u = w♭
        # Update nlp to stay on manifold
        ExaPF.update!(pen, u)
        # Compute objective
        c = ExaPF.objective(pen, u)
        c_ref = ExaPF.inner_objective(pen, u)
        @test isa(c, Real)
        @test c >= c_ref
        inf_pr2 = ExaPF.primal_infeasibility(pen, u)
        @test inf_pr2 >= 0.0

        ##################################################
        # Update penalty weigth
        # (with a large-enough factor to have a meaningful derivative check)
        ##################################################
        ExaPF.update_penalty!(pen, η=1e3)
        ExaPF.update_multipliers!(pen)

        ##################################################
        # Callbacks
        ##################################################
        ExaPF.update!(pen, u)
        obj = ExaPF.objective(pen, u)
        g = ExaPF.gradient(pen, u)
        # Compare with finite differences
        function reduced_cost(u_)
            ExaPF.update!(pen, u_)
            return ExaPF.objective(pen, u_)
        end
        grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, u)
        h_grad_fd = grad_fd[:] |> Array
        h_g = g |> Array
        @test isapprox(h_grad_fd, h_g, rtol=1e-5)

        # Test Hessian only on ReducedSpaceEvaluator and SlackEvaluator
        if (
           isa(nlp, ExaPF.ReducedSpaceEvaluator) ||
           (isa(nlp, ExaPF.SlackEvaluator) && isa(device, CPU)) # Currently not supported because of Jacobian!
        )
            n = length(u)
            ExaPF.update!(pen, u)
            hv = similar(u) ; fill!(hv, 0)
            w = similar(u)
            h_w = zeros(n) ; h_w[1] = 1.0
            copyto!(w, h_w)

            ExaPF.hessprod!(pen, hv, u, w)
            H = similar(u, n, n) ; fill!(H, 0)
            ExaPF.hessian!(pen, H, u)
            # Is Hessian vector product relevant?
            @test H * w ≈ hv
            # Is Hessian correct?
            hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, u)

            h_H = H |> Array
            h_H_fd = hess_fd.data

            @test isapprox(h_H, h_H_fd, rtol=1e-5)
        end
        # Test estimation of multipliers (only on SlackEvaluator)
        if isa(nlp, ExaPF.SlackEvaluator) && isa(device, CPU)
            λ = ExaPF.estimate_multipliers(pen, u)
        end
    end
end

