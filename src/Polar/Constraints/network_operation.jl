
# Lagrangian
is_constraint(::typeof(network_operations)) = true
size_constraint(polar::PolarForm, ::typeof(network_operations)) = 2 * get(polar, PS.NumberOfBuses()) + 1

KA.@kernel function _bus_operation_kernel!(
    cons, pnet,
    @Const(pinj), @Const(qinj), @Const(pload), @Const(qload),
    @Const(pv), @Const(pq), @Const(ref), @Const(pv_to_gen), @Const(ref_to_gen),
)
    i, j = @index(Global, NTuple)

    npv = length(pv)
    npq = length(pq)
    nref = length(ref)
    nbus = npv + npq + nref

    #= PQ NODE =#
    if i <= npq
        bus = pq[i]
        # Balance
        cons[i+npv, j]     = pinj[bus, j] + pload[bus, j]
        cons[i+npv+npq, j] = qinj[bus, j] + qload[bus, j]

    #= PV NODE =#
    elseif i <= npq + npv
        i_ = i - npq
        bus = pv[i_]
        i_gen = pv_to_gen[i_]
        # Balance
        cons[i_, j] = pinj[bus, j] - pnet[bus, j] + pload[bus, j]
        # Reactive power generation
        shift = npv + 2 * npq + nref
        cons[i_gen + shift, j] = qinj[bus, j] + qload[bus, j]

    #= REF NODE =#
    elseif i <= npq + npv + nref
        i_ = i - npv - npq
        bus = ref[i_]
        i_gen = ref_to_gen[i_]

        # Active power generation
        shift = npv + 2 * npq
        pg = pinj[bus, j] + pload[bus, j]
        cons[i_ + shift, j] = pg
        pnet[bus, j] = pg
        # Reactive power generation
        shift = npv + 2 * npq + nref
        cons[i_gen + shift, j] = qinj[bus, j] + qload[bus, j]
    end
end

KA.@kernel function _bcost_kernel!(
    costs, @Const(pnet), @Const(coefs),
    @Const(pv), @Const(ref), @Const(pv_to_gen), @Const(ref_to_gen),
)
    i, j = @index(Global, NTuple)
    npv = length(pv)
    nref = length(ref)
    # Evaluate active power at PV nodes
    if i <= npv
        bus = pv[i]
        i_gen = pv_to_gen[i]
    # Evaluate active power at slack nodes
    elseif i <= npv + nref
        i_ = i - npv
        bus = ref[i_]
        i_gen = ref_to_gen[i_]
    end

    pg = pnet[bus, j]
    c0 = coefs[i_gen, 2]
    c1 = coefs[i_gen, 3]
    c2 = coefs[i_gen, 4]
    costs[i_gen, j] = quadratic_cost(pg, c0, c1, c2)
end

function network_operations(
    polar::PolarForm{T, VI, VT, MT}, cons, vmag, vang, pnet, qnet, pd, qd
) where {T, VI, VT, MT}
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    pv_to_gen = polar.indexing.index_pv_to_gen
    ref_to_gen = polar.indexing.index_ref_to_gen

    nbatch = size(cons, 2)

    fill!(cons, 0.0)
    injection = MT(undef, 2 * nbus, nbatch)
    fill!(injection, 0.0)

    # Compute injection
    bus_power_injection(polar, injection, vmag, vang, pnet, qnet, pd, qd)

    pinj = view(injection, 1:nbus, :)
    qinj = view(injection, 1+nbus:2*nbus, :)

    # Compute operations
    ndrange = (nbus, nbatch)
    # Constraints
    ev = _bus_operation_kernel!(polar.device)(
        cons, pnet, pinj, qinj, pd, qd,
        pv, pq, ref, pv_to_gen, ref_to_gen,
        ndrange=ndrange, dependencies=Event(polar.device),
    )
    wait(ev)

    # Objective
    coefs = polar.costs_coefficients
    costs = similar(cons, ngen, nbatch)
    ev = _bcost_kernel!(polar.device)(
        costs, pnet, coefs, pv, ref, pv_to_gen, ref_to_gen,
        ndrange=(ngen, nbatch), dependencies=Event(polar.device),
    )
    wait(ev)

    cons[end, :] .= sum(costs)
    return
end

function network_operations(polar::PolarForm, cons::AbstractVector, buffer::PolarNetworkState)
    network_operations(polar, cons, buffer.vmag, buffer.vang, buffer.pnet, buffer.qnet, buffer.pload, buffer.qload)
end

function AutoDiff.TapeMemory(
    polar::PolarForm, func::typeof(network_operations), VT; with_stack=true,
    nbatch=1,
)
    nnz = length(polar.topology.ybus_im.nzval)
    nx = get(polar, NumberOfState())
    nbus = get(polar, PS.NumberOfBuses())
    # Intermediate state
    intermediate = (
        ∂inj = VT(undef, 2*nbus),
        ∂edge_vm_fr = VT(undef, nnz),
        ∂edge_va_fr = VT(undef, nnz),
        ∂edge_vm_to = VT(undef, nnz),
        ∂edge_va_to = VT(undef, nnz),
    )
    return AutoDiff.TapeMemory(
        network_operations,
        (with_stack) ? AdjointPolar(polar) : nothing,
        intermediate,
    )
end

KA.@kernel function _adjoint_bus_operation_kernel!(
    adj_inj, adj_pnet,
    @Const(adj_op), @Const(pnet),
    @Const(c0), @Const(c1), @Const(c2),
    @Const(pv), @Const(pq), @Const(ref), @Const(pv_to_gen), @Const(ref_to_gen),
)
    i, j = @index(Global, NTuple)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)
    nbus = npv + npq + nref

    #= PQ NODE =#
    if i <= npq
        bus = pq[i]
        # Injection
        adj_inj[bus     , j] = adj_op[i+npv]      # wrt P
        adj_inj[bus+nbus, j] = adj_op[i+npv+npq]  # wrt Q

    #= PV NODE =#
    elseif i <= npq + npv
        i_ = i - npq
        bus = pv[i_]
        i_gen = pv_to_gen[i_]
        # Generation
        pg = pnet[bus, j]
        adj_pnet[bus, j] = adj_op[end] * adj_quadratic_cost(pg, c0[i_gen], c1[i_gen], c2[i_gen])
        # Active injection
        adj_inj[bus, j] = adj_op[i_]  # wrt P
        # Reactive injection
        shift = npv + 2 * npq + nref
        adj_inj[bus + nbus, j] = adj_op[i_gen + shift]  # wrt Q

    #= REF NODE =#
    elseif i <= npq + npv + nref
        i_ = i - npv - npq
        bus = ref[i_]
        i_gen = ref_to_gen[i_]

        pg = pnet[bus, j]
        adj_pg = adj_op[end] * adj_quadratic_cost(pg, c0[i_gen], c1[i_gen], c2[i_gen])
        adj_pnet[bus, j] = adj_pg

        shift = npv + 2 * npq
        adj_inj[bus, j] = adj_op[i_+shift] + adj_pg

        shift = npv + 2 * npq + nref
        adj_inj[bus + nbus, j] = adj_op[i_gen+shift]
    end
end

function _adjoint_network_operations(polar::PolarForm, ∂inj, ∂pnet, ∂cons, pnet)
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    pq = polar.indexing.index_pq
    pv = polar.indexing.index_pv
    ref = polar.indexing.index_ref
    pv2gen = polar.indexing.index_pv_to_gen
    ref2gen = polar.indexing.index_ref_to_gen

    coefs = polar.costs_coefficients
    c0 = @view coefs[:, 2]
    c1 = @view coefs[:, 3]
    c2 = @view coefs[:, 4]

    ndrange = (nbus, size(pnet, 2))
    _adjoint_bus_operation_kernel!(polar.device)(
        ∂inj, ∂pnet, ∂cons, pnet, c0, c1, c2, pv, pq, ref, pv2gen, ref2gen,
        ndrange=ndrange, dependencies=Event(polar.device),
    )
end

function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    cons, ∂cons,
    vm, ∂vm,
    va, ∂va,
    pnet, ∂pnet,
    pload, qload,
) where {F<:typeof(network_operations), S, I}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    pv = polar.indexing.index_pv
    ref = polar.indexing.index_ref
    pv2gen = polar.indexing.index_pv_to_gen
    ref2gen = polar.indexing.index_ref_to_gen

    # Intermediate state
    ∂inj = pbm.intermediate.∂inj

    fill!(∂vm, 0.0)
    fill!(∂va, 0.0)
    fill!(∂pnet, 0.0)

    # Seed adjoint of injection
    _adjoint_network_operations(polar, ∂inj, ∂pnet, ∂cons, pnet)
    # Backpropagate through the power injection to get ∂vm and ∂va
    _adjoint_bus_power_injection!(polar, pbm, ∂inj, vm, ∂vm, va, ∂va)
    return
end

