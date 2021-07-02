is_constraint(::typeof(cost_production)) = true
size_constraint(polar::PolarForm, ::typeof(cost_production)) = 1

function pullback_objective(polar::PolarForm)
    return AutoDiff.TapeMemory(
        cost_production,
        AdjointStackObjective(polar),
        nothing,
    )
end

@inline quadratic_cost(pg, c0, c1, c2) = c0 + c1 * pg + c2 * pg^2
@inline adj_quadratic_cost(pg, c0, c1, c2) = c1 + 2.0 * c2 * pg

KA.@kernel function cost_production_kernel!(
    costs, pg, @Const(vmag), @Const(vang), pnet, @Const(pload),
    @Const(c0), @Const(c1), @Const(c2),
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
        pg[i_gen, j] = pnet[bus, j]
    # Evaluate active power at slack nodes
    elseif i <= npv + nref
        i_ = i - npv
        bus = ref[i_]
        i_gen = ref_to_gen[i_]
        inj = bus_injection(bus, j, vmag, vang, ybus_re_colptr, ybus_re_rowval, ybus_re_nzval, ybus_im_nzval)
        pg[i_gen, j] = inj + pload[bus]
        pnet[bus, j] = inj + pload[bus]
    end

    costs[i_gen, j] = quadratic_cost(pg[i_gen, j], c0[i_gen], c1[i_gen], c2[i_gen])
end

KA.@kernel function adj_cost_production_kernel!(
    adj_costs,
    @Const(vmag), adj_vmag, @Const(vang), adj_vang, @Const(pnet), adj_pnet, @Const(pload),
    @Const(c0), @Const(c1), @Const(c2),
    @Const(pv), @Const(ref), @Const(pv_to_gen), @Const(ref_to_gen),
    @Const(ybus_re_nzval), @Const(ybus_re_colptr), @Const(ybus_re_rowval), @Const(ybus_im_nzval),
)
    i, j = @index(Global, NTuple)
    npv = length(pv)
    nref = length(ref)
    if i <= npv
        bus = pv[i]
        i_gen = pv_to_gen[i]
        pg = pnet[bus, j]
        adj_pnet[bus, j] = adj_costs[1] * adj_quadratic_cost(pg, c0[i_gen], c1[i_gen], c2[i_gen])
    # Evaluate active power at slack nodes
    elseif i <= npv + nref
        i_ = i - npv
        bus = ref[i_]
        i_gen = ref_to_gen[i_]

        inj = bus_injection(bus, j, vmag, vang, ybus_re_colptr, ybus_re_rowval, ybus_re_nzval, ybus_im_nzval)
        pg = inj + pload[bus]

        adj_net = adj_costs[1] * adj_quadratic_cost(pg, c0[i_gen], c1[i_gen], c2[i_gen])
        adj_pnet[bus, j] = adj_net
        adjoint_bus_injection!(
            bus, j, adj_net, adj_vmag, adj_vang, vmag, vang, ybus_re_colptr, ybus_re_rowval, ybus_re_nzval, ybus_im_nzval
        )
    end
end

function cost_production(polar::PolarForm, buffer::PolarNetworkState)
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

    ev = cost_production_kernel!(polar.device)(
        costs, buffer.pgen,
        buffer.vmag, buffer.vang, buffer.pnet, buffer.pload,
        c0, c1, c2,
        pv, ref, pv2gen, ref2gen,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval, ybus_im.nzval,
        ndrange=(ngen, size(buffer.pgen, 2)),
        dependencies=Event(polar.device)
    )
    wait(ev)
    # TODO: supports batch
    return sum(costs)
end

function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    pg, ∂cost,
    vm, ∂vm,
    va, ∂va,
    pnet, ∂pnet,
    pload, qload,
) where {F<:typeof(cost_production), S, I}
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
    fill!(∂pnet, 0.0)
    ev = adj_cost_production_kernel!(polar.device)(
        ∂cost,
        vm, ∂vm,
        va, ∂va,
        pnet, ∂pnet, pload,
        c0, c1, c2,
        index_pv, index_ref, pv2gen, ref2gen,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval, ybus_im.nzval,
        ndrange=(ngen, size(∂vm, 2)),
        dependencies=Event(polar.device)
    )
    wait(ev)
    return
end

function gradient_objective!(polar::PolarForm, ∂obj::AutoDiff.TapeMemory, buffer::PolarNetworkState)
    ∂pg = ∂obj.stack.∂pg
    obj_autodiff = ∂obj.stack
    adj_pg = obj_autodiff.∂pg
    adj_x = obj_autodiff.∇fₓ
    adj_u = obj_autodiff.∇fᵤ
    adj_vmag = obj_autodiff.∂vm
    adj_vang = obj_autodiff.∂va
    adj_pinj = obj_autodiff.∂pinj

    # Adjoint of active power generation
    adjoint!(polar, ∂obj,
        buffer.pgen, 1.0,
        buffer.vmag, adj_vmag,
        buffer.vang, adj_vang,
        buffer.pnet, adj_pinj,
        buffer.pload, buffer.qload,
    )

    # Adjoint w.r.t. x and u
    fill!(adj_x, 0.0)
    fill!(adj_u, 0.0)
    adjoint_transfer!(polar, adj_u, adj_x, adj_vmag, adj_vang, adj_pinj)
    return
end

