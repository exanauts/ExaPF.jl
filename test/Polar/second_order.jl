function test_hessprod_with_finitediff(polar, device, MT; rtol=1e-6, atol=1e-6)
    nx = ExaPF.number(polar, State())
    nu = ExaPF.number(polar, Control())

    mymap = [ExaPF.mapping(polar, State()); ExaPF.mapping(polar, Control())]

    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    # Need to copy structures on the host for FiniteDiff.jl
    polar_cpu = ExaPF.PolarForm(polar, CPU())
    stack_cpu = ExaPF.NetworkStack(polar_cpu)
    basis_cpu  = ExaPF.PolarBasis(polar_cpu)

    # Solve power flow
    ExaPF.run_pf(polar, stack)
    copyto!(stack_cpu.input, stack.input)

    # Tests all expressions in once with MultiExpressions
    constraints = [
        ExaPF.VoltageMagnitudeBounds(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.LineFlows(polar),
        ExaPF.PowerFlowBalance(polar),
    ]
    mycons = ExaPF.MultiExpressions(constraints) ∘ basis

    constraints_cpu = [
        ExaPF.VoltageMagnitudeBounds(polar_cpu),
        ExaPF.PowerGenerationBounds(polar_cpu),
        ExaPF.LineFlows(polar_cpu),
        ExaPF.PowerFlowBalance(polar_cpu),
    ]
    mycons_cpu = ExaPF.MultiExpressions(constraints_cpu) ∘ basis_cpu

    # Initiate state and control for FiniteDiff
    # CONSTRAINTS
    m = length(mycons)
    μ = rand(m)
    c = zeros(m)

    HessianAD = ExaPF.HessianProd(polar, mycons, mymap)
    tgt = rand(nx + nu)
    projp = zeros(nx + nu)
    dev_tgt = MT(tgt)
    dev_projp = MT(projp)
    dev_μ = MT(μ)
    ExaPF.hprod!(HessianAD, dev_projp, stack, dev_μ, dev_tgt)
    projp = Array(dev_projp)

    ∂stack_cpu = ExaPF.NetworkStack(polar_cpu)
    empty!(∂stack_cpu)
    function grad_lagr_x(z)
        stack_cpu.input[mymap] .= z
        mycons_cpu(c, stack_cpu)
        empty!(∂stack_cpu)
        ExaPF.adjoint!(mycons_cpu, ∂stack_cpu, stack_cpu, μ)
        return ∂stack_cpu.input[mymap]
    end
    x0 = stack_cpu.input[mymap]
    H_fd = FiniteDiff.finite_difference_jacobian(grad_lagr_x, x0)
    proj_fd = zeros(nx+nu)
    mul!(proj_fd, H_fd, tgt)

    @test myisapprox(projp, proj_fd, rtol=rtol)
end

function test_full_space_hessian(polar, device, MT)
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    # Need to copy structures on the host for FiniteDiff.jl
    polar_cpu = ExaPF.PolarForm(polar, CPU())
    stack_cpu = ExaPF.NetworkStack(polar_cpu)
    basis_cpu  = ExaPF.PolarBasis(polar_cpu)

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

    constraints_cpu = [
        ExaPF.CostFunction(polar_cpu),
        ExaPF.PowerFlowBalance(polar_cpu),
        ExaPF.VoltageMagnitudeBounds(polar_cpu),
        ExaPF.PowerGenerationBounds(polar_cpu),
        ExaPF.LineFlows(polar_cpu),
    ]
    mycons_cpu = ExaPF.MultiExpressions(constraints_cpu) ∘ basis_cpu

    m = length(mycons)
    y_cpu = rand(m)
    y = y_cpu |> MT

    hess = ExaPF.FullHessian(polar, mycons, mymap)
    H = ExaPF.hessian!(hess, stack, y)

    c = zeros(m)
    ∂stack_cpu = ExaPF.NetworkStack(polar_cpu)

    function grad_fd_x(x)
        stack_cpu.input[mymap] .= x
        mycons_cpu(c, stack_cpu)
        empty!(∂stack_cpu)
        ExaPF.adjoint!(mycons_cpu, ∂stack_cpu, stack_cpu, y_cpu)
        return ∂stack_cpu.input[mymap]
    end
    x = stack_cpu.input[mymap]
    Hd = FiniteDiff.finite_difference_jacobian(grad_fd_x, x)

    # Test that both Hessian match
    @test myisapprox(Hd, H, rtol=1e-5)
    return
end

function test_block_hessian(polar, device, MT)
    nblocks = 3
    mapx = ExaPF.mapping(polar, State())

    # Single evaluation
    stack = ExaPF.NetworkStack(polar)
    mycons = ExaPF.MultiExpressions([
        ExaPF.CostFunction(polar),
        ExaPF.PowerFlowBalance(polar),
        ExaPF.VoltageMagnitudeBounds(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.LineFlows(polar),
    ]) ∘ ExaPF.PolarBasis(polar)
    m = length(mycons)
    y = ones(m) |> MT
    hess = ExaPF.FullHessian(polar, mycons, mapx)
    # Eval!
    H = ExaPF.hessian!(hess, stack, y)

    # Block evaluation
    blk_polar = ExaPF.BlockPolarForm(polar, nblocks)
    blk_stack = ExaPF.NetworkStack(blk_polar)
    blk_cons = ExaPF.MultiExpressions([
        ExaPF.CostFunction(blk_polar),
        ExaPF.PowerFlowBalance(blk_polar),
        ExaPF.VoltageMagnitudeBounds(blk_polar),
        ExaPF.PowerGenerationBounds(blk_polar),
        ExaPF.LineFlows(blk_polar),
    ]) ∘ ExaPF.PolarBasis(blk_polar)
    blk_y = repeat(ones(m), nblocks) |> MT
    blk_hess = ExaPF.ArrowheadHessian(blk_polar, blk_cons, ExaPF.State())
    # Eval!
    blk_H = ExaPF.hessian!(blk_hess, blk_stack, blk_y)

    # Test results match
    blk_H_cpu = blk_H |> SparseMatrixCSC
    H_cpu = H |> SparseMatrixCSC
    @test blk_H_cpu ≈ blockdiag([H_cpu for i in 1:nblocks]...)

    # Multivariables
    for X in [State(), Control(), AllVariables()]
        blk_hess = ExaPF.ArrowheadHessian(blk_polar, blk_cons, X)
        blk_H = ExaPF.hessian!(blk_hess, blk_stack, blk_y)
        @test isa(blk_H, AbstractMatrix)
    end
end

