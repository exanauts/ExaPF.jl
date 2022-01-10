function test_hessian_with_matpower(polar, device, AT; atol=1e-6, rtol=1e-6)
    pf = polar.network
    pv = pf.pv ; npv = length(pv)
    pq = pf.pq ; npq = length(pq)
    ref = pf.ref ; nref = length(ref)
    nbus = pf.nbus
    # Cache
    cache = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, cache)
    # Jacobian AD
    jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
    ju = AutoDiff.Jacobian(polar, ExaPF.power_balance, Control())
    ∂obj = ExaPF.AdjointStackObjective(polar)

    conv = powerflow(polar, jx, cache, NewtonRaphson())

    ##################################################
    # Computation of Hessians
    ##################################################
    @testset "Compare with Matpower's Hessian ($constraints)" for constraints in [
        ExaPF.power_balance,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints,
    ]
        ncons = ExaPF.size_constraint(polar, constraints)
        hλ = rand(ncons)
        λ = hλ |> AT
        # Evaluate Hessian-vector product (full ∇²g is a 3rd dimension tensor)
        ∇²gλ = ExaPF.matpower_hessian(polar, constraints, cache, hλ)
        nx = size(∇²gλ.xx, 1)
        nu = size(∇²gλ.uu, 1)

        # Hessian-vector product using forward over adjoint AD
        HessianAD = AutoDiff.Hessian(polar, constraints)

        projp = zeros(nx + nu) |> AT

        host_tgt = rand(nx + nu)
        tgt = host_tgt |> AT
        tgt[nx+1:end] .= 0.0
        AutoDiff.adj_hessian_prod!(polar, HessianAD, projp, cache, λ, tgt)
        host_projp = projp |> Array
        @test isapprox(host_projp[1:nx], ∇²gλ.xx * host_tgt[1:nx])

        host_tgt = rand(nx + nu)
        tgt = host_tgt |> AT
        # set tangents only for u direction
        tgt[1:nx] .= 0.0
        AutoDiff.adj_hessian_prod!(polar, HessianAD, projp, cache, λ, tgt)
        host_projp = projp |> Array
        # (we use absolute tolerance as Huu is equal to 0 for case9)
        @test isapprox(host_projp[nx+1:end], ∇²gλ.uu * host_tgt[nx+1:end], atol=atol)

        # check cross terms ux
        host_tgt = rand(nx + nu)
        tgt = host_tgt |> AT
        # Build full Hessian
        H = [
            ∇²gλ.xx ∇²gλ.xu';
            ∇²gλ.xu ∇²gλ.uu
        ]
        AutoDiff.adj_hessian_prod!(polar, HessianAD, projp, cache, λ, tgt)
        host_projp = projp |> Array
        @test isapprox(host_projp, H * host_tgt)
    end

    return nothing
end

function test_hessprod_with_finitediff(polar, device, MT; rtol=1e-6, atol=1e-6)
    nx = length(polar.mapx)
    nu = length(polar.mapu)

    mymap = [ExaPF.my_map(polar, State()); ExaPF.my_map(polar, Control())]

    stack = ExaPF.NetworkStack(polar)
    ExaPF.forward_eval_intermediate(polar, stack)
    # Solve power flow
    conv = ExaPF.run_pf(polar, stack)

    # Tests all expressions in once with MultiExpressions
    constraints = [
        ExaPF.VoltageMagnitudePQ(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.LineFlows(polar),
        ExaPF.PowerFlowBalance(polar),
    ]
    mycons = ExaPF.MultiExpressions(constraints)

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
        ExaPF.forward_eval_intermediate(polar, stack)
        mycons(c, stack)
        return dot(μ, c)
    end
    x0 = stack.input[mymap]
    H_fd = FiniteDiff.finite_difference_hessian(lagr_x, x0)
    proj_fd = similar(x0, nx+nu)
    mul!(proj_fd, H_fd.data, dev_tgt, 1, 0)

    @test isapprox(projp, Array(proj_fd), rtol=rtol)
end

function test_full_space_hessian(polar, device, MT)
    stack = ExaPF.NetworkStack(polar)
    ExaPF.forward_eval_intermediate(polar, stack)

    n = length(stack.input)
    # Hessian / (x, u)
    mymap = [ExaPF.my_map(polar, State()); ExaPF.my_map(polar, Control())]

    constraints = [
        # ExaPF.CostFunction(polar),
        ExaPF.PowerFlowBalance(polar),
        ExaPF.PowerGenerationBounds(polar),
        ExaPF.LineFlows(polar),
    ]
    mycons = ExaPF.MultiExpressions(constraints)

    m = length(mycons)
    y = rand(m) |> MT

    hess = ExaPF.FullHessian(polar, mycons, mymap)
    H = ExaPF.hessian!(hess, stack, y)
    c = zeros(m) |> MT

    function hess_fd_x(x)
        stack.input[mymap] .= x
        ExaPF.forward_eval_intermediate(polar, stack)
        mycons(c, stack)
        return dot(c, y)
    end
    x = stack.input[mymap]
    Hd = FiniteDiff.finite_difference_hessian(hess_fd_x, x)
    @test myisapprox(Hd.data, H, rtol=1e-5)
    return
end

