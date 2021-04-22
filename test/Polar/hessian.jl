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
    pbm = AutoDiff.TapeMemory(ExaPF.active_power_generation, ∂obj, nothing)

    conv = powerflow(polar, jx, cache, NewtonRaphson())
    ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)

    ##################################################
    # Computation of Hessians
    ##################################################
    @testset "Compare with Matpower's Hessian ($constraints)" for constraints in [
        ExaPF.power_balance,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints,
    ]
        ncons = ExaPF.size_constraint(polar, constraints)
        λ = rand(ncons) |> AT
        # Evaluate Hessian-vector product (full ∇²g is a 3rd dimension tensor)
        ∇²gλ = ExaPF.matpower_hessian(polar, constraints, cache, λ)
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

function test_hessian_with_finitediff(polar, device, MT; rtol=1e-6, atol=1e-6)
    pf = polar.network
    pv = pf.pv ; npv = length(pv)
    pq = pf.pq ; npq = length(pq)
    ref = pf.ref ; nref = length(ref)
    nbus = pf.nbus
    ngen = get(polar, PS.NumberOfGenerators())

    pv2gen = polar.indexing.index_pv_to_gen
    ref2gen = polar.indexing.index_ref_to_gen
    gen2bus = polar.indexing.index_generators
    cache = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, cache)

    xk = ExaPF.initial(polar, State())
    u = ExaPF.initial(polar, Control())
    nx = length(xk) ; nu = length(u)

    jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
    ju = AutoDiff.Jacobian(polar, ExaPF.power_balance, Control())
    ∂obj = ExaPF.AdjointStackObjective(polar)
    pbm = AutoDiff.TapeMemory(ExaPF.active_power_generation, ∂obj, nothing)

    # Initiate state and control for FiniteDiff
    x = [cache.vang[pv] ; cache.vang[pq] ; cache.vmag[pq]]
    u = [cache.vmag[ref]; cache.vmag[pv]; cache.pg[pv2gen]]

    @testset "Compare with FiniteDiff Hessian ($constraints)" for constraints in [
        ExaPF.power_balance,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints,
        ExaPF.flow_constraints,
    ]
        ncons = ExaPF.size_constraint(polar, constraints)
        μ = rand(ncons)

        function jac_x(z)
            x_ = z[1:nx]
            u_ = z[1+nx:end]
            # Transfer control
            ExaPF.transfer!(polar, cache, u_)
            # Transfer state (manually)
            cache.vang[pv] .= x_[1:npv]
            cache.vang[pq] .= x_[npv+1:npv+npq]
            cache.vmag[pq] .= x_[npv+npq+1:end]
            ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)
            Jx = ExaPF.matpower_jacobian(polar, constraints, State(), cache)
            Ju = ExaPF.matpower_jacobian(polar, constraints, Control(), cache)
            return [Jx Ju]' * μ
        end
        H_fd = FiniteDiff.finite_difference_jacobian(jac_x, [x; u])

        HessianAD = AutoDiff.Hessian(polar, constraints)
        tgt = rand(nx + nu)
        projp = zeros(nx + nu)
        dev_tgt = MT(tgt)
        dev_projp = MT(projp)
        dev_μ = MT(μ)
        AutoDiff.adj_hessian_prod!(polar, HessianAD, dev_projp, cache, dev_μ, dev_tgt)
        projp = Array(dev_projp)
        @test isapprox(projp, H_fd * tgt, rtol=rtol)
    end
end

