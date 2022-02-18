function test_constraints_jacobian(polar, device, MT)
    nx = ExaPF.number(polar, State())
    nu = ExaPF.number(polar, Control())

    stack = ExaPF.NetworkStack(polar)
    ∂stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    mymap = [ExaPF.mapping(polar, State()); ExaPF.mapping(polar, Control())]

    # Solve power flow
    conv = ExaPF.run_pf(polar, stack)
    # Get solution in complex form.
    V = ExaPF.voltage_host(stack)

    # Test Jacobian w.r.t. State
    @testset "Jacobian $(expr)" for expr in [
        ExaPF.PolarBasis,
        ExaPF.VoltageMagnitudeBounds,
        ExaPF.PowerFlowBalance,
        ExaPF.PowerGenerationBounds,
        ExaPF.LineFlows,
    ]
        @testset "Colorings" for coloring in [
            ExaPF.AutoDiff.SparseDiffToolsColoring(),
            ExaPF.AutoDiff.ColPackColoring(),
        ]

            constraint = expr(polar) ∘ basis
            m = length(constraint)

            # Allocation
            jac = ExaPF.Jacobian(polar, constraint, mymap; coloring=coloring)
            # Test display
            println(devnull, jac)
            # Evaluate Jacobian with AD
            J = ExaPF.jacobian!(jac, stack)
            # Matpower Jacobian
            Jmat = ExaPF.matpower_jacobian(polar, constraint, V)
            Jmat = Jmat[:, mymap]

            # Compare with FiniteDiff
            function jac_fd_x(x)
                stack.input[mymap] .= x
                c = zeros(m) |> MT
                constraint(c, stack)
                return c
            end
            x = copy(stack.input[mymap])
            Jd = FiniteDiff.finite_difference_jacobian(jac_fd_x, x) |> Array
            Jx = jac.J |> SparseMatrixCSC |> Array

            ## JACOBIAN VECTOR PRODUCT
            tgt_h = rand(m)
            tgt = tgt_h |> MT
            output = zeros(nx+nu) |> MT
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
end

function test_constraints_adjoint(polar, device, MT)
    nx = ExaPF.number(polar, State())
    nu = ExaPF.number(polar, Control())
    mymap = [ExaPF.mapping(polar, State()); ExaPF.mapping(polar, Control())]

    stack = ExaPF.NetworkStack(polar)
    ∂stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    conv = ExaPF.run_pf(polar, stack)

    @testset "Adjoint $(expr)" for expr in [
        ExaPF.PolarBasis,
        ExaPF.CostFunction,
        ExaPF.VoltageMagnitudeBounds,
        ExaPF.PowerFlowBalance,
        ExaPF.PowerGenerationBounds,
        ExaPF.LineFlows,
    ]
        constraint = expr(polar) ∘ basis
        m = length(constraint)
        tgt = rand(m) |> MT
        output = zeros(nx+nu) |> MT

        c = zeros(m) |> MT
        constraint(c, stack)

        empty!(∂stack)
        ExaPF.adjoint!(constraint, ∂stack, stack, tgt)
        function test_fd(x)
            stack.input[mymap] .= x
            constraint(c, stack)
            return dot(c, tgt)
        end
        x = copy(stack.input[mymap])
        adj_fd = FiniteDiff.finite_difference_jacobian(test_fd, x) |> Array
        # Loosen the tolerance to 1e-5 there (finite_difference_jacobian
        # is less accurate than finite_difference_gradient)
        @test myisapprox(∂stack.input[mymap], adj_fd[:], rtol=1e-5)
    end
end

function test_full_space_jacobian(polar, device, MT)
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    n = length(stack.input)
    mymap = collect(1:n)

    constraints = [
        ExaPF.VoltageMagnitudeBounds(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.LineFlows(polar),
        ExaPF.PowerFlowBalance(polar),
    ]
    mycons = ExaPF.MultiExpressions(constraints) ∘ basis

    m = length(mycons)

    jac = ExaPF.Jacobian(polar, mycons, mymap)
    J = ExaPF.jacobian!(jac, stack)

    function jac_fd_x(x)
        stack.input .= x
        c = zeros(m) |> MT
        mycons(c, stack)
        return c
    end
    x = copy(stack.input)
    Jd = FiniteDiff.finite_difference_jacobian(jac_fd_x, x) |> Array
    @test myisapprox(Jd, J, rtol=1e-5)
end

function test_reduced_gradient(polar, device, MT)
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)
    ∂stack = ExaPF.NetworkStack(polar)

    power_balance = ExaPF.PowerFlowBalance(polar) ∘ basis

    mapx = ExaPF.mapping(polar, State())
    mapu = ExaPF.mapping(polar, Control())
    nx = length(mapx)
    nu = length(mapu)

    jx = ExaPF.Jacobian(polar, power_balance, mapx)
    ju = ExaPF.Jacobian(polar, power_balance, mapu)

    # Solve power flow
    solver = NewtonRaphson(tol=1e-12)
    ExaPF.nlsolve!(solver, jx, stack)
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

    c = zeros(1)
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
        stack.input[mapu] .= u_
        ExaPF.nlsolve!(solver, jx, stack)
        cost_production(c, stack)
        return c[1]
    end

    u = stack.input[mapu]
    grad_fd = FiniteDiff.finite_difference_jacobian(reduced_cost, u)
    @test isapprox(grad_fd[:], grad_adjoint, rtol=1e-4)
end

