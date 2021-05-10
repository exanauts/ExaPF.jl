is_constraint(::typeof(cost_penalty_ramping_constraints)) = true
size_constraint(polar::PolarForm, ::typeof(cost_penalty_ramping_constraints)) = 1

function pullback_ramping(polar::PolarForm, intermediate)
    return AutoDiff.TapeMemory(
        cost_penalty_ramping_constraints,
        AdjointStackObjective(polar),
        intermediate,
    )
end

@inline function _cost_ramping(pg, s, c0, c1, c2, σ, t, τ, λf, λt, ρf, ρt, p1, p2, p3)
    obj = σ * quadratic_cost(pg, c0, c1, c2)
    penalty = 0.5 * τ * (pg - p2)^2
    if t != 0
        penalty += λf * (p1 - pg + s) + 0.5 * ρf * (p1 - pg + s)^2
    end
    if t != 1
        penalty += λt * (pg - p3) + 0.5 * ρt * (pg - p3)^2
    end
    return obj + penalty
end

@inline function _adjoint_cost_ramping(pg, s, c0, c1, c2, σ, t, τ, λf, λt, ρf, ρt, p1, p2, p3)
    ∂c = σ * adj_quadratic_cost(pg, c0, c1, c2)
    ∂c += τ * (pg - p2)
    if t != 0
       ∂c -= λf + ρf * (p1 - pg + s)
    end
    if t != 1
       ∂c += λt + ρt * (pg - p3)
    end
    return ∂c
end

KA.@kernel function cost_ramping_kernel!(
    costs, pg, @Const(vmag), @Const(vang), @Const(pinj), @Const(pload), @Const(s),
    @Const(c0), @Const(c1), @Const(c2),
    @Const(σ), @Const(t), @Const(τ), @Const(λf), @Const(λt), @Const(ρf), @Const(ρt), @Const(p1), @Const(p2), @Const(p3),
    @Const(pv), @Const(ref), @Const(pv_to_gen), @Const(ref_to_gen),
    @Const(ybus_re_nzval), @Const(ybus_re_colptr), @Const(ybus_re_rowval),
    @Const(ybus_im_nzval),
)
    i, j = @index(Global, NTuple)
    npv = length(pv)
    nref = length(ref)
    # Evaluate active power at PV nodes
    if i <= npv
        bus = pv[i]
        i_gen = pv_to_gen[i]
        pg[i_gen, j] = pinj[bus, j]
    # Evaluate active power at slack nodes
    elseif i <= npv + nref
        i_ = i - npv
        bus = ref[i_]
        i_gen = ref_to_gen[i_]
        inj = bus_injection(bus, j, vmag, vang, ybus_re_colptr, ybus_re_rowval, ybus_re_nzval, ybus_im_nzval)
        pg[i_gen, j] = inj + pload[bus]
    end

    costs[i_gen, j] = _cost_ramping(
        pg[i_gen, j], s[i_gen, j], c0[i_gen], c1[i_gen], c2[i_gen],
        σ, t, τ, λf[i_gen], λt[i_gen], ρf, ρt, p1[i_gen], p2[i_gen], p3[i_gen],
    )
end

KA.@kernel function adj_cost_ramping_kernel!(
    adj_costs,
    @Const(vmag), adj_vmag, @Const(vang), adj_vang, @Const(pinj), adj_pinj, @Const(pload),
    @Const(s),
    @Const(c0), @Const(c1), @Const(c2),
    @Const(σ), @Const(t), @Const(τ), @Const(λf), @Const(λt), @Const(ρf), @Const(ρt), @Const(p1), @Const(p2), @Const(p3),
    @Const(pv), @Const(ref), @Const(pv_to_gen), @Const(ref_to_gen),
    @Const(ybus_re_nzval), @Const(ybus_re_colptr), @Const(ybus_re_rowval), @Const(ybus_im_nzval),
)
    i, j = @index(Global, NTuple)
    npv = length(pv)
    nref = length(ref)
    if i <= npv
        bus = pv[i]
        i_gen = pv_to_gen[i]
        pg = pinj[bus, j]
        adj_pinj[bus, j] = adj_costs[1] * _adjoint_cost_ramping(
            pg, s[i_gen], c0[i_gen], c1[i_gen], c2[i_gen],
            σ, t, τ, λf[i_gen], λt[i_gen], ρf, ρt, p1[i_gen], p2[i_gen], p3[i_gen],
        )
    # Evaluate active power at slack nodes
    elseif i <= npv + nref
        i_ = i - npv
        fr = ref[i_]
        i_gen = ref_to_gen[i_]

        inj = bus_injection(fr, j, vmag, vang, ybus_re_colptr, ybus_re_rowval, ybus_re_nzval, ybus_im_nzval)
        pg = inj + pload[fr]

        adj_inj = adj_costs[1] * _adjoint_cost_ramping(
            pg, s[i_gen], c0[i_gen], c1[i_gen], c2[i_gen],
            σ, t, τ, λf[i_gen], λt[i_gen], ρf, ρt, p1[i_gen], p2[i_gen], p3[i_gen],
        )
        adj_pinj[fr, j] = adj_inj
        # Update adj_vmag, adj_vang
        adjoint_bus_injection!(
            fr, j, adj_inj, adj_vmag, adj_vang, vmag, vang, ybus_re_colptr, ybus_re_rowval, ybus_re_nzval, ybus_im_nzval
        )
    end
end

function cost_penalty_ramping_constraints(
    polar::PolarForm, buffer::PolarNetworkState,
    s, t, σ, τ, λf, λt, ρf, ρt, p1, p2, p3,
)
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    pv2gen = polar.indexing.index_pv_to_gen
    ref2gen = polar.indexing.index_ref_to_gen
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    ngen = PS.get(polar, PS.NumberOfGenerators())
    coefs = polar.costs_coefficients
    c0 = @view coefs[:, 2]
    c1 = @view coefs[:, 3]
    c2 = @view coefs[:, 4]
    costs = similar(buffer.pgen)

    ev = cost_ramping_kernel!(polar.device)(
        costs, buffer.pgen,
        buffer.vmag, buffer.vang, buffer.pnet, buffer.pload, s,
        c0, c1, c2,
        σ, t, τ, λf, λt, ρf, ρt, p1, p2, p3,
        pv, ref, pv2gen, ref2gen,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval, ybus_im.nzval,
        ndrange=(ngen, size(buffer.pgen, 2)),
        dependencies=Event(polar.device),
    )
    wait(ev)
    return sum(costs)
end

function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    pg, ∂cost,
    vm, ∂vm,
    va, ∂va,
    pinj, ∂pinj,
    pload, qload,
) where {F<:typeof(cost_penalty_ramping_constraints), S, I}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    index_pv = polar.indexing.index_pv
    index_ref = polar.indexing.index_ref
    pv2gen = polar.indexing.index_pv_to_gen
    ref2gen = polar.indexing.index_ref_to_gen

    coefs = polar.costs_coefficients
    c0 = @view coefs[:, 2]
    c1 = @view coefs[:, 3]
    c2 = @view coefs[:, 4]

    ngen = get(polar, PS.NumberOfGenerators())
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    fill!(∂vm, 0.0)
    fill!(∂va, 0.0)
    fill!(∂pinj, 0.0)
    ev = adj_cost_ramping_kernel!(polar.device)(
        ∂cost,
        vm, ∂vm,
        va, ∂va,
        pinj, ∂pinj, pload, pbm.intermediate.s,
        c0, c1, c2,
        pbm.intermediate.σ,
        pbm.intermediate.t,
        pbm.intermediate.τ,
        pbm.intermediate.λf,
        pbm.intermediate.λt,
        pbm.intermediate.ρf,
        pbm.intermediate.ρt,
        pbm.intermediate.p1,
        pbm.intermediate.p2,
        pbm.intermediate.p3,
        index_pv, index_ref, pv2gen, ref2gen,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval, ybus_im.nzval,
        ndrange=(ngen, size(∂vm, 2)),
        dependencies=Event(polar.device)
    )
    wait(ev)
    return
end

