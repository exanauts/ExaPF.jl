function test_hessprod_with_finitediff(polar, device, MT; rtol=1e-6, atol=1e-6)
    nx = ExaPF.number(polar, State())
    nu = ExaPF.number(polar, Control())

    mymap = [ExaPF.my_map(polar, State()); ExaPF.my_map(polar, Control())]

    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    # Solve power flow
    conv = ExaPF.run_pf(polar, stack)

    # Tests all expressions in once with MultiExpressions
    constraints = [
        ExaPF.VoltageMagnitudePQ(polar),
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

    HessianAD = ExaPF.MyHessian(polar, mycons, mymap)
    tgt = rand(nx + nu)
    projp = zeros(nx + nu)
    dev_tgt = MT(tgt)
    dev_projp = MT(projp)
    dev_μ = MT(μ)
    ExaPF.hprod!(HessianAD, dev_projp, stack, dev_μ, dev_tgt)
    projp = Array(dev_projp)

    function lagr_x(z)
        stack.input[mymap] .= z
        mycons(c, stack)
        return dot(μ, c)
    end
    x0 = stack.input[mymap]
    H_fd = FiniteDiff.finite_difference_hessian(lagr_x, x0)
    proj_fd = similar(x0, nx+nu)
    mul!(proj_fd, H_fd.data, dev_tgt, 1, 0)

    @test myisapprox(projp, Array(proj_fd), rtol=rtol)
end

function test_full_space_hessian(polar, device, MT)
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    n = length(stack.input)
    # Hessian / (x, u)
    mymap = [ExaPF.my_map(polar, State()); ExaPF.my_map(polar, Control())]

    constraints = [
        # ExaPF.CostFunction(polar),
        ExaPF.PowerFlowBalance(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.LineFlows(polar),
    ]
    mycons = ExaPF.MultiExpressions(constraints) ∘ basis

    m = length(mycons)
    y = rand(m) |> MT

    hess = ExaPF.FullHessian(polar, mycons, mymap)
    H = ExaPF.hessian!(hess, stack, y)
    c = zeros(m) |> MT

    function hess_fd_x(x)
        stack.input[mymap] .= x
        mycons(c, stack)
        return dot(c, y)
    end
    x = stack.input[mymap]
    Hd = FiniteDiff.finite_difference_hessian(hess_fd_x, x)
    @test myisapprox(Hd.data, H, rtol=1e-5)
    return
end

