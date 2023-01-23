function test_constraints_jacobian(polar, device, MT)
    nx = ExaPF.number(polar, State())
    nu = ExaPF.number(polar, Control())

    stack = ExaPF.NetworkStack(polar)
    ∂stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    # Need to copy structures on the host for FiniteDiff.jl
    polar_cpu = ExaPF.PolarForm(polar, CPU())
    stack_cpu = ExaPF.NetworkStack(polar_cpu)
    basis_cpu  = ExaPF.PolarBasis(polar_cpu)

    mymap = [ExaPF.mapping(polar, State()); ExaPF.mapping(polar, Control())]

    # Solve power flow
    ExaPF.run_pf(polar, stack)
    ExaPF.run_pf(polar_cpu, stack_cpu)
    # Get solution in complex form.
    V = ExaPF.voltage_host(stack)

    # Test Jacobian w.r.t. State
    @testset "Jacobian $(expr)" for expr in [
        ExaPF.PolarBasis,
        # ExaPF.VoltageMagnitudeBounds,
        # ExaPF.PowerFlowBalance,
        # ExaPF.PowerGenerationBounds,
        # ExaPF.LineFlows,
    ]
        constraint = expr(polar) ∘ basis
        constraint_cpu = expr(polar_cpu) ∘ basis_cpu
        m = length(constraint)

        c_ = zeros(m) |> MT
        constraint(c_, stack)
        # Allocation
        jac = ExaPF.Jacobian(polar, constraint, mymap)
        # Test display
        println(devnull, jac)
        # Evaluate Jacobian with AD
        J = ExaPF.jacobian!(jac, stack)
        # Matpower Jacobian
        Jmat = ExaPF.matpower_jacobian(polar, constraint, V)
        Jmat = Jmat[:, mymap]

        # Compare with FiniteDiff
        function jac_fd_x(x)
            stack_cpu.input[mymap] .= x
            c = zeros(m)
            constraint_cpu(c, stack_cpu)
            return c
        end
        x = copy(stack_cpu.input[mymap])
        Jd = FiniteDiff.finite_difference_jacobian(jac_fd_x, x)
        Jx = jac.J |> SparseMatrixCSC |> Array

        ## JACOBIAN VECTOR PRODUCT
        tgt_h = rand(m)
        tgt = tgt_h |> MT
        empty!(∂stack)
        ExaPF.adjoint!(constraint, ∂stack, stack, tgt)

        @test size(J) == (m, length(mymap))
        @test myisapprox(Jd, Jx, rtol=1e-5)
        @test myisapprox(Jmat, Jx, rtol=1e-5)
        @test myisapprox(Jmat, Jd, rtol=1e-5)
        @test myisapprox(∂stack.input[mymap], Jx' * tgt_h, rtol=1e-6)
        @test myisapprox(∂stack.input[mymap], Jmat' * tgt_h, rtol=1e-6)
    end
end

function test_constraints_adjoint(polar, device, MT)
    nx = ExaPF.number(polar, State())
    nu = ExaPF.number(polar, Control())
    mymap = [ExaPF.mapping(polar, State()); ExaPF.mapping(polar, Control())]

    stack = ExaPF.NetworkStack(polar)
    ∂stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    # Need to copy structures on the host for FiniteDiff.jl
    polar_cpu = ExaPF.PolarForm(polar, CPU())
    stack_cpu = ExaPF.NetworkStack(polar_cpu)
    basis_cpu  = ExaPF.PolarBasis(polar_cpu)

    ExaPF.run_pf(polar, stack)
    ExaPF.run_pf(polar_cpu, stack_cpu)

    @testset "Adjoint $(expr)" for expr in [
        ExaPF.PolarBasis,
        # ExaPF.CostFunction,
        # ExaPF.VoltageMagnitudeBounds,
        # ExaPF.PowerFlowBalance,
        # ExaPF.PowerGenerationBounds,
        # ExaPF.LineFlows,
    ]
        # constraint = expr(polar) ∘ basis
        # constraint_cpu = expr(polar_cpu) ∘ basis_cpu
        constraint = expr(polar)
        constraint_cpu = expr(polar_cpu)
        m = length(constraint)
        tgt_cpu = rand(m)
        tgt = tgt_cpu |> MT

        c_ = zeros(m) |> MT
        constraint(c_, stack)

        empty!(∂stack)
        ExaPF.adjoint!(constraint, ∂stack, stack, tgt)

        function test_fd(x)
            stack_cpu.input[mymap] .= x
            c = zeros(m)
            constraint_cpu(c, stack_cpu)
            return dot(c, tgt_cpu)
        end
        x = copy(stack_cpu.input[mymap])
        adj_fd = FiniteDiff.finite_difference_gradient(test_fd, x)
        @test myisapprox(∂stack.input[mymap], adj_fd[:], rtol=1e-6)
    end
end

function test_full_space_jacobian(polar, device, MT)
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    # Need to copy structures on the host for FiniteDiff.jl
    polar_cpu = ExaPF.PolarForm(polar, CPU())
    stack_cpu = ExaPF.NetworkStack(polar_cpu)
    basis_cpu  = ExaPF.PolarBasis(polar_cpu)

    n = length(stack.input)
    mymap = collect(1:n)

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

    m = length(mycons)

    jac = ExaPF.Jacobian(polar, mycons, mymap)
    J = ExaPF.jacobian!(jac, stack)

    function jac_fd_x(x)
        stack_cpu.input .= x
        c = zeros(m)
        mycons_cpu(c, stack_cpu)
        return c
    end
    x = copy(stack_cpu.input)
    Jd = FiniteDiff.finite_difference_jacobian(jac_fd_x, x)
    @test myisapprox(Jd, J, rtol=1e-5)
end

function test_reduced_gradient(polar, device, MT)
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)
    ∂stack = ExaPF.NetworkStack(polar)

    # Need to copy structures on the host for FiniteDiff.jl
    polar_cpu = ExaPF.PolarForm(polar, CPU())
    stack_cpu = ExaPF.NetworkStack(polar_cpu)
    basis_cpu  = ExaPF.PolarBasis(polar_cpu)

    power_balance = ExaPF.PowerFlowBalance(polar) ∘ basis
    power_balance_cpu = ExaPF.PowerFlowBalance(polar_cpu) ∘ basis_cpu

    mapx = ExaPF.mapping(polar, State())
    mapu = ExaPF.mapping(polar, Control())
    nx = length(mapx)
    nu = length(mapu)

    jx = ExaPF.Jacobian(polar, power_balance, mapx)
    ju = ExaPF.Jacobian(polar, power_balance, mapu)

    jx_cpu = ExaPF.Jacobian(polar_cpu, power_balance_cpu, mapx)

    # Solve power flow
    solver = NewtonRaphson(tol=1e-12)
    ExaPF.nlsolve!(solver, jx, stack)

    copyto!(stack_cpu.input, stack.input)

    # No need to recompute ∇gₓ
    ∇gₓ = jx.J
    ∇gᵤ = ExaPF.jacobian!(ju, stack)

    # Test with Matpower's Jacobian
    V = ExaPF.voltage_host(stack)
    J = ExaPF.matpower_jacobian(polar, power_balance, V)
    h∇gₓ = ∇gₓ |> SparseMatrixCSC |> Array
    h∇gᵤ = ∇gᵤ |> SparseMatrixCSC |> Array
    @test isapprox(h∇gₓ, J[:, mapx])
    @test isapprox(h∇gᵤ, J[:, mapu])

    cost_production = ExaPF.CostFunction(polar) ∘ basis
    cost_production_cpu = ExaPF.CostFunction(polar_cpu) ∘ basis_cpu

    c = zeros(1) |> MT
    cost_production(c, stack)

    grad = similar(stack.input, nx+nu)

    empty!(∂stack)
    ExaPF.adjoint!(cost_production, ∂stack, stack, 1.0)
    ∇fₓ = ∂stack.input[mapx]
    ∇fᵤ = ∂stack.input[mapu]

    h∇fₓ = ∇fₓ |> Array
    h∇fᵤ = ∇fᵤ |> Array
    ## ADJOINT
    # lamba calculation
    λk  = -(h∇gₓ') \ h∇fₓ
    grad_adjoint = h∇fᵤ + h∇gᵤ' * λk
    # ## DIRECT
    S = - inv(h∇gₓ) * h∇gᵤ
    grad_direct = h∇fᵤ + S' * h∇fₓ
    @test isapprox(grad_adjoint, grad_direct)

    # Compare with finite difference
    function reduced_cost(u_)
        stack_cpu.input[mapu] .= u_
        ExaPF.nlsolve!(solver, jx_cpu, stack_cpu)
        c_ = zeros(1)
        cost_production_cpu(c_, stack_cpu)
        return sum(c_)
    end

    u = stack_cpu.input[mapu]
    grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, u)
    @test isapprox(grad_fd[:], grad_adjoint, rtol=1e-6)
end

function test_block_jacobian(polar, device, MT)
    nblocks = 3
    mapx = ExaPF.mapping(polar, State())

    stack = ExaPF.NetworkStack(polar)
    blk_stack = ExaPF.BlockNetworkStack(polar, nblocks)

    for expr in [
        ExaPF.PowerFlowBalance(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.VoltageMagnitudeBounds(polar),
        ExaPF.LineFlows(polar),
    ]
        pf = expr ∘ ExaPF.PolarBasis(polar)
        m = length(pf)

        jac = ExaPF.Jacobian(polar, pf, mapx)
        blk_jac = ExaPF.ArrowheadJacobian(polar, pf, State(), nblocks)

        ExaPF.jacobian!(jac, stack)
        ExaPF.jacobian!(blk_jac, blk_stack)

        blk_J_cpu = blk_jac.J |> SparseMatrixCSC
        J_cpu = jac.J |> SparseMatrixCSC
        @test blk_J_cpu ≈ blockdiag([J_cpu for i in 1:nblocks]...)
    end

    constraints = [
        ExaPF.PowerFlowBalance(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.VoltageMagnitudeBounds(polar),
        ExaPF.LineFlows(polar),
    ]
    mycons = ExaPF.MultiExpressions(constraints) ∘ ExaPF.PolarBasis(polar)
    for X in [State(), Control(), AllVariables()]
        blk_jac = ExaPF.ArrowheadJacobian(polar, mycons, X, nblocks)
        ExaPF.jacobian!(blk_jac, blk_stack)
    end
end

