function test_constraints_jacobian(polar, device, MT)
    nx = ExaPF.number(polar, State())
    nu = ExaPF.number(polar, Control())

    stack = ExaPF.NetworkStack(polar)
    ∂stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    mymap = [ExaPF.my_map(polar, State()); ExaPF.my_map(polar, Control())]

    # Solve power flow
    conv = ExaPF.run_pf(polar, stack)
    # Get solution in complex form.
    V = ExaPF.voltage_host(stack)

    # Test Jacobian w.r.t. State
    @testset "Jacobian $(expr)" for expr in [
        ExaPF.PolarBasis,
        ExaPF.VoltageMagnitudePQ,
        ExaPF.PowerFlowBalance,
        ExaPF.PowerGenerationBounds,
        ExaPF.LineFlows,
    ]
        constraint = expr(polar) ∘ basis
        m = length(constraint)

        # Allocation

        jac = ExaPF.MyJacobian(polar, constraint, mymap)
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
        @test isapprox(∂stack.input[mymap], Jx' * tgt_h, rtol=1e-6)
        @test isapprox(∂stack.input[mymap], Jmat' * tgt_h, rtol=1e-6)
    end
end

function test_constraints_adjoint(polar, device, MT)
    nx = ExaPF.number(polar, State())
    nu = ExaPF.number(polar, Control())
    mymap = [ExaPF.my_map(polar, State()); ExaPF.my_map(polar, Control())]

    stack = ExaPF.NetworkStack(polar)
    ∂stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    conv = ExaPF.run_pf(polar, stack)

    @testset "Adjoint $(expr)" for expr in [
        ExaPF.PolarBasis,
        ExaPF.CostFunction,
        ExaPF.VoltageMagnitudePQ,
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
        @test isapprox(∂stack.input[mymap], adj_fd[:], rtol=1e-5)
    end
end

function test_full_space_jacobian(polar, device, MT)
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    n = length(stack.input)
    mymap = collect(1:n)

    constraints = [
        ExaPF.VoltageMagnitudePQ(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.LineFlows(polar),
        ExaPF.PowerFlowBalance(polar),
    ]
    mycons = ExaPF.MultiExpressions(constraints) ∘ basis

    m = length(mycons)

    jac = ExaPF.MyJacobian(polar, mycons, mymap)
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

