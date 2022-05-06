function test_hessprod_with_finitediff(polar, device, MT; rtol=1e-6, atol=1e-6)
    nx = ExaPF.number(polar, State())
    nu = ExaPF.number(polar, Control())

    mymap = [ExaPF.mapping(polar, State()); ExaPF.mapping(polar, Control())]

    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    # Solve power flow
    conv = ExaPF.run_pf(polar, stack)

    # Tests all expressions in once with MultiExpressions
    constraints = [
        ExaPF.VoltageMagnitudeBounds(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.LineFlows(polar),
        ExaPF.PowerFlowBalance(polar),
    ]
    mycons = ExaPF.MultiExpressions(constraints) ∘ basis

    # Initiate state and control for FiniteDiff
    # CONSTRAINTS
    m = length(mycons)
    μ = rand(m) |> MT
    c = zeros(m) |> MT

    HessianAD = ExaPF.HessianProd(polar, mycons, mymap)
    tgt = rand(nx + nu)
    projp = zeros(nx + nu)
    dev_tgt = MT(tgt)
    dev_projp = MT(projp)
    dev_μ = MT(μ)
    ExaPF.hprod!(HessianAD, dev_projp, stack, dev_μ, dev_tgt)
    projp = Array(dev_projp)

    ∂stack = ExaPF.NetworkStack(polar)
    empty!(∂stack)
    function grad_lagr_x(z)
        stack.input[mymap] .= z
        mycons(c, stack)
        empty!(∂stack)
        ExaPF.adjoint!(mycons, ∂stack, stack, dev_μ)
        return ∂stack.input[mymap]
    end
    x0 = stack.input[mymap]
    H_fd = FiniteDiff.finite_difference_jacobian(grad_lagr_x, x0)
    proj_fd = similar(x0, nx+nu)
    mul!(proj_fd, H_fd, dev_tgt, 1, 0)

    @test myisapprox(projp, Array(proj_fd), rtol=rtol)
end

function test_full_space_hessian(polar, device, MT)
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    n = length(stack.input)
    # Hessian / (x, u)
    mymap = [ExaPF.mapping(polar, State()); ExaPF.mapping(polar, Control())]

    constraints = [
        ExaPF.CostFunction(polar),
        ExaPF.PowerFlowBalance(polar),
        ExaPF.VoltageMagnitudeBounds(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.LineFlows(polar),
    ]
    mycons = ExaPF.MultiExpressions(constraints) ∘ basis

    m = length(mycons)
    y_cpu = rand(m)
    y = y_cpu |> MT

    hess = ExaPF.FullHessian(polar, mycons, mymap)
    H = ExaPF.hessian!(hess, stack, y)

    c = zeros(m) |> MT
    ∂stack = ExaPF.NetworkStack(polar)

    function grad_fd_x(x)
        stack.input[mymap] .= x
        mycons(c, stack)
        empty!(∂stack)
        ExaPF.adjoint!(mycons, ∂stack, stack, y)
        return ∂stack.input[mymap]
    end
    x = stack.input[mymap]
    Hd = FiniteDiff.finite_difference_jacobian(grad_fd_x, x)

    # Test that both Hessian match
    @test myisapprox(Hd, H, rtol=1e-5)
    return
end

function test_batch_hessian(polar, device, MT)
    nblocks = 3
    mapx = ExaPF.mapping(polar, State())

    stack = ExaPF.NetworkStack(polar)
    blk_stack = ExaPF.BlockNetworkStack(polar, nblocks)

    basis  = ExaPF.PolarBasis(polar)
    constraints = [
        ExaPF.CostFunction(polar),
        ExaPF.PowerFlowBalance(polar),
        ExaPF.VoltageMagnitudeBounds(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.LineFlows(polar),
    ]
    mycons = ExaPF.MultiExpressions(constraints) ∘ basis

    m = length(mycons)
    y = ones(m) |> MT
    blk_y = repeat(ones(m), nblocks) |> MT

    # Evaluate reference Hessian
    hess = ExaPF.FullHessian(polar, mycons, mapx)
    H = ExaPF.hessian!(hess, stack, y)
    # Block evaluation
    blk_hess = ExaPF.ArrowheadHessian(polar, mycons, ExaPF.State(), nblocks)
    blk_H = ExaPF.hessian!(blk_hess, blk_stack, blk_y)
    blk_H_cpu = blk_H |> SparseMatrixCSC
    H_cpu = H |> SparseMatrixCSC
    @test blk_H_cpu ≈ blockdiag([H_cpu for i in 1:nblocks]...)

    # Multivariables
    for X in [State(), Control(), AllVariables()]
        blk_hess = ExaPF.ArrowheadHessian(polar, mycons, X, nblocks)
        blk_H = ExaPF.hessian!(blk_hess, blk_stack, blk_y)
        @test isa(blk_H, AbstractMatrix)
    end
end

