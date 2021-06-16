
function test_proxal_evaluator(nlp, device, MT)
    u0 = ExaPF.initial(nlp)
    @testset "ProxALEvaluators ($time)" for time in [ExaPF.Origin, ExaPF.Normal, ExaPF.Final]
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
            @test_broken H ≈ hess_fd.data rtol=1e-6
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
            if isa(device, CPU)
                rows, cols = ExaPF.jacobian_structure(prox)
                # Evaluation
                jac = ExaPF.jacobian(prox, w)
                # Check correctness of transpose Jacobian vector product
                @test jv == jac' * v

                # Jacobian vector product
                ExaPF.jprod!(prox, v, w, jv)
                @test v == jac * jv
            end
        end

        ExaPF.reset!(prox)
    end
end
