function test_reduced_gradient(polar, device, MT)
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)
    ∂stack = ExaPF.NetworkStack(polar)

    power_balance = ExaPF.PowerFlowBalance(polar) ∘ basis

    mapx = ExaPF.my_map(polar, State())
    mapu = ExaPF.my_map(polar, Control())
    nx = length(mapx)
    nu = length(mapu)

    jx = ExaPF.MyJacobian(polar, power_balance, mapx)
    ju = ExaPF.MyJacobian(polar, power_balance, mapu)

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

