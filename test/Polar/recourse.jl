
function test_recourse_powerflow(polar, device, M)
    k = 2
    polar_ext = ExaPF.PolarFormRecourse(polar, k)
    stack = ExaPF.NetworkStack(polar_ext)

    pf_recourse = ExaPF.PowerFlowRecourse(polar_ext) ∘ ExaPF.PolarBasis(polar_ext)
    jac_recourse = ExaPF.ArrowheadJacobian(polar_ext, pf_recourse, State())
    ExaPF.set_params!(jac_recourse, stack)
    ExaPF.jacobian!(jac_recourse, stack)

    pf_solver = NewtonRaphson()
    convergence = ExaPF.nlsolve!(
        pf_solver,
        jac_recourse,
        stack,
    )
    @test convergence.has_converged
    @test convergence.norm_residuals < pf_solver.tol

    residual = pf_recourse(stack)
    @test norm(residual) < pf_solver.tol
    return
end

function test_recourse_expression(polar, device, M)
    k = 2
    polar_ext = ExaPF.PolarFormRecourse(polar, k)
    @test ExaPF.nblocks(polar_ext) == k

    stack = ExaPF.NetworkStack(polar_ext)

    ngen = PS.get(polar, PS.NumberOfGenerators())
    @test polar_ext.ncustoms == k * (ngen + 1)
    @test length(stack.vuser) == k * (ngen + 1)

    for expr in [
        ExaPF.PowerFlowRecourse,
        ExaPF.ReactivePowerBounds,
    ]
        ev = expr(polar_ext) ∘ ExaPF.PolarBasis(polar_ext)
        res = ev(stack)
        @test isa(res, M)
        @test length(res) == length(ev)

        g_min, g_max = ExaPF.bounds(polar_ext, ev)
        @test length(g_min) == length(ev)
        @test length(g_max) == length(ev)
        @test isa(g_min, M)
        @test isa(g_max, M)
        @test myisless(g_min, g_max)
    end
    return
end

function test_recourse_jacobian(polar, device, M)
    k = 2
    polar_ext = ExaPF.PolarFormRecourse(polar, k)
    stack = ExaPF.NetworkStack(polar_ext)
    ∂stack = ExaPF.NetworkStack(polar_ext)
    stack_fd = ExaPF.NetworkStack(polar_ext)

    nu = ExaPF.number(polar_ext, Control())
    mapx = ExaPF.mapping(polar_ext, State(), k)
    mapu = ExaPF.mapping(polar_ext, Control(), k)
    mapxu = ExaPF.mapping(polar_ext, AllVariables(), k)

    for expr in [
        ExaPF.PowerFlowRecourse,
        ExaPF.ReactivePowerBounds,
    ]
        ev = expr(polar_ext) ∘ ExaPF.PolarBasis(polar_ext)
        m = length(ev)

        # Compute ref with finite-diff
        function _fd_func(x)
            stack_fd.input .= x
            return ev(stack_fd)
        end
        x0 = copy(stack.input)
        Jd = FiniteDiff.finite_difference_jacobian(_fd_func, x0)
        Jd_x = Jd[:, mapx]
        Jd_u = sum(Jd[:, mapu[1+(i-1)*nu:i*nu]] for i in 1:k)
        Jd_xu = [Jd_x Jd_u]

        # / State
        jac_x = ExaPF.ArrowheadJacobian(polar_ext, ev, State())
        Jx = ExaPF.jacobian!(jac_x, stack) |> SparseMatrixCSC
        @test Jx ≈ Jd_x rtol=1e-5

        # / Control
        jac_u = ExaPF.ArrowheadJacobian(polar_ext, ev, Control())
        Ju = ExaPF.jacobian!(jac_u, stack) |> SparseMatrixCSC
        @test Ju ≈ Jd_u rtol=1e-5

        # / all
        jac_xu = ExaPF.ArrowheadJacobian(polar_ext, ev, AllVariables())
        Jxu = ExaPF.jacobian!(jac_xu, stack) |> SparseMatrixCSC
        @test Jxu ≈ Jd_xu rtol=1e-5

        # Test adjoint
        tgt_h = rand(m)
        tgt = tgt_h |> M
        empty!(∂stack)
        ExaPF.adjoint!(ev, ∂stack, stack, tgt)
        # Compare with finite diff
        function _fd_adj(x)
            stack_fd.input .= x
            return dot(tgt_h, ev(stack_fd))
        end
        adj_fd = FiniteDiff.finite_difference_gradient(_fd_adj, x0)
        @test myisapprox(∂stack.input[mapxu], adj_fd[mapxu], rtol=1e-6)
        @test myisapprox(∂stack.input[mapxu], Jd[:, mapxu]' * tgt_h, rtol=1e-6)
    end
    return
end

function test_recourse_hessian(polar, device, M)
    k = 1
    polar_ext = ExaPF.PolarFormRecourse(polar, k)
    stack = ExaPF.NetworkStack(polar_ext)
    stack_fd = ExaPF.NetworkStack(polar_ext)
    ∂stack = ExaPF.NetworkStack(polar_ext)

    basis  = ExaPF.PolarBasis(polar_ext)
    mapxu = ExaPF.mapping(polar_ext, AllVariables())

    constraints = [
        ExaPF.QuadraticCost(polar_ext),
        ExaPF.PowerFlowRecourse(polar_ext),
        ExaPF.ReactivePowerBounds(polar_ext),
        ExaPF.LineFlows(polar_ext),
    ]
    ev = ExaPF.MultiExpressions(constraints) ∘ basis

    m = length(ev)
    c = zeros(m)
    y = rand(m)

    # Evaluate Hessian
    hess = ExaPF.FullHessian(polar_ext, ev, mapxu)
    H = ExaPF.hessian!(hess, stack, y)

    # Evaluate Hessian with finite-diff
    function _fd_hess(x)
        stack_fd.input[mapxu] .= x
        return dot(y, ev(stack_fd))
    end
    x0 = stack.input[mapxu]
    Hd = FiniteDiff.finite_difference_hessian(_fd_hess, x0)
    @test myisapprox(H, Hd, rtol=1e-5)
    return
end

function test_recourse_block_hessian(polar, device, M)
    k = 2
    polar_ext = ExaPF.PolarFormRecourse(polar, k)
    stack = ExaPF.NetworkStack(polar_ext)
    stack_fd = ExaPF.NetworkStack(polar_ext)
    ∂stack = ExaPF.NetworkStack(polar_ext)

    basis  = ExaPF.PolarBasis(polar_ext)
    mapxu = ExaPF.mapping(polar_ext, State(), k)

    constraints = [
        ExaPF.PowerFlowRecourse(polar_ext),
        ExaPF.ReactivePowerBounds(polar_ext),
        ExaPF.LineFlows(polar_ext),
    ]
    ev = ExaPF.MultiExpressions(constraints) ∘ basis

    m = length(ev)
    c = zeros(m)
    y = rand(m)

    # Evaluate Hessian
    hess = ExaPF.ArrowheadHessian(polar_ext, ev, AllVariables())
    H = ExaPF.hessian!(hess, stack, y)

    # Evaluate Hessian with finite-diff
    function _fd_grad(x)
        stack_fd.input[mapxu] .= x
        ev(c, stack_fd)
        empty!(∂stack)
        ExaPF.adjoint!(ev, ∂stack, stack_fd, y)
        return ∂stack.input[mapxu]
    end
    x0 = stack.input[mapxu]
    Hd = FiniteDiff.finite_difference_jacobian(_fd_grad, x0)

    # Test evaluation matches with finite-diff
    # TODO
    # @test myisapprox(H, Hd, rtol=1e-5)
    return H, Hd
end

