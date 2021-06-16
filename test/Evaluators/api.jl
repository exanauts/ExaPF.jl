function test_evaluator_api(nlp, device, M)
    # Test printing
    println(devnull, nlp)

    n = ExaPF.n_variables(nlp)
    m = ExaPF.n_constraints(nlp)
    u = ExaPF.initial(nlp)

    u_min, u_max = ExaPF.bounds(nlp, ExaPF.Variables())
    g_min, g_max = ExaPF.bounds(nlp, ExaPF.Constraints())
    buffer = get(nlp, ExaPF.PhysicalState())

    # Test consistence
    @test n == length(u)
    @test length(u_min) == length(u_max) == n
    @test isless(u_min, u_max)
    @test length(g_min) == length(g_max) == m
    if m > 0
        @test g_min <= g_max
    end

    # Test API
    @test isa(get(nlp, State()), AbstractVector)
    @test isa(get(nlp, ExaPF.Constraints()), Array{Function})
    @test isa(get(nlp, State()), AbstractVector)
    @test isa(buffer, ExaPF.AbstractBuffer)
    @test ExaPF.constraints_type(nlp) in [:bound, :equality, :inequality]

    @test isa(ExaPF.has_hessian(nlp), Bool)
    @test isa(ExaPF.has_hessian_lagrangian(nlp), Bool)

    # setters
    nbus = get(nlp, PS.NumberOfBuses())
    loads = similar(u, nbus) ; fill!(loads, 1)
    ExaPF.setvalues!(nlp, PS.ActiveLoad(), loads)
    ExaPF.setvalues!(nlp, PS.ReactiveLoad(), loads)

    ExaPF.reset!(nlp)
end

function test_evaluator_callbacks(nlp, device, M; rtol=1e-6)
    n = ExaPF.n_variables(nlp)
    m = ExaPF.n_constraints(nlp)
    u = ExaPF.initial(nlp)

    u_min, u_max = ExaPF.bounds(nlp, ExaPF.Variables())
    g_min, g_max = ExaPF.bounds(nlp, ExaPF.Constraints())

    # 1/ update! function
    conv = ExaPF.update!(nlp, u)
    @test isa(conv, ExaPF.ConvergenceStatus)
    @test conv.has_converged

    # 2/ objective function
    c = ExaPF.objective(nlp, u)
    @test isa(c, Real)

    # 3/ gradient! function
    function reduced_cost(u_)
        ExaPF.update!(nlp, u_)
        return ExaPF.objective(nlp, u_)
    end
    g = similar(u) ; fill!(g, 0)
    ExaPF.gradient!(nlp, g, u)
    grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, u)
    @test isapprox(grad_fd, g, rtol=rtol)

    # Constraint
    # 4/ Constraint
    ## Evaluation of the constraints
    if m > 0
        cons = similar(g_min) ; fill!(cons, 0)
        ExaPF.constraint!(nlp, cons, u)

        ## Evaluation of the transpose-Jacobian product
        jv = similar(u_min) ; fill!(jv, 0.0)
        v = similar(g_min) ; fill!(v, 1.0)
        ExaPF.jtprod!(nlp, jv, u, v)
        function reduced_cons(u_)
            ExaPF.update!(nlp, u_)
            ExaPF.constraint!(nlp, cons, u_)
            return dot(v, cons)
        end
        jv_fd = FiniteDiff.finite_difference_gradient(reduced_cons, u)

        # TODO: rtol=1e-6 breaks on case30. Investigate why.
        @test isapprox(jv, jv_fd, rtol=1e-4)

        ## Evaluation of the Jacobian (only on CPU)
        J = ExaPF.jacobian(nlp, u)
        # Test transpose Jacobian vector product
        @test isapprox(jv, J' * v)
        # Test Jacobian vector product
        ExaPF.jprod!(nlp, v, u, jv)
        @test isapprox(J * jv, v)
    end

    ExaPF.reset!(nlp)
end

function test_evaluator_hessian(nlp, device, M; rtol=1e-6)
    n = ExaPF.n_variables(nlp)
    @test ExaPF.has_hessian(nlp)
    function reduced_cost(u_)
        ExaPF.update!(nlp, u_)
        return ExaPF.objective(nlp, u_)
    end
    u = ExaPF.initial(nlp)
    ExaPF.update!(nlp, u)
    ExaPF.gradient(nlp, u) # compute the gradient to update the adjoint internally

    # 1/ Hessian-vector product
    hv = similar(u) ; fill!(hv, 0)
    w = similar(u) ; fill!(w, 0)
    w[1] = 1.0
    ExaPF.hessprod!(nlp, hv, u, w)

    # 2/ Full Hessian
    H = similar(u, n, n) ; fill!(H, 0)
    ExaPF.hessian!(nlp, H, u)

    # 3/ FiniteDiff
    hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, u)

    @test H * w == hv
    @test H ≈ hess_fd.data rtol=rtol
end

function test_evaluator_batch_hessian(nlp, device, M; rtol=1e-5)
    n = ExaPF.n_variables(nlp)
    nbatch = ExaPF.number_batches_hessian(nlp)
    @test ExaPF.has_hessian(nlp)
    @test nbatch > 1
    function reduced_cost(u_)
        ExaPF.update!(nlp, u_)
        return ExaPF.objective(nlp, u_)
    end

    u = ExaPF.initial(nlp)
    n = length(u)
    ExaPF.update!(nlp, u)
    g = ExaPF.gradient(nlp, u) # compute the gradient to update the adjoint internally

    # 0/ Update Hessian object
    # 1/ Hessian-vector product
    hv = similar(u, n, nbatch) ; fill!(hv, 0)
    w = similar(u, n, nbatch) ; fill!(w, 0)
    w[1, :] .= 1.0
    ExaPF.hessprod!(nlp, hv, u, w)

    # 2/ Full Hessian
    H = similar(u, n, n) ; fill!(H, 0)
    ExaPF.hessian!(nlp, H, u)

    # 3/ FiniteDiff
    hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, u)

    @test H * w == hv
    @test H ≈ hess_fd.data rtol=rtol

    m = ExaPF.n_constraints(nlp)
    if m > 0
        J = similar(u, m, n)
        ExaPF.jacobian!(nlp, J, u)
        function reduced_cons(u_)
            cons = similar(u_, m)
            ExaPF.update!(nlp, u_)
            ExaPF.constraint!(nlp, cons, u_)
            return cons
        end
        J_fd = FiniteDiff.finite_difference_jacobian(reduced_cons, u)
        @test J ≈ J_fd rtol=rtol
    end
end

