# Adjoints needed in polar formulation
#

"""
    AdjointStackObjective{VT}

An object for storing the adjoint stack in the adjoint objective computation.

"""
struct AdjointStackObjective{VT}
    ∇fₓ::VT
    ∇fᵤ::VT
    ∂pg::VT
    ∂vm::VT
    ∂va::VT
    jvₓ::VT
    jvᵤ::VT
    ∂flow::VT
end

function put_active_power_injection!(fr, v_m, v_a, adj_v_m, adj_v_a, adj_P, ybus_re, ybus_im)
    @inbounds for c in ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1
        to = ybus_re.rowval[c]
        aij = v_a[fr] - v_a[to]
        cθ = ybus_re.nzval[c]*cos(aij)
        sθ = ybus_im.nzval[c]*sin(aij)
        adj_v_m[fr] += v_m[to] * (cθ + sθ) * adj_P
        adj_v_m[to] += v_m[fr] * (cθ + sθ) * adj_P

        adj_aij = -(v_m[fr]*v_m[to]*(ybus_re.nzval[c]*sin(aij)))
        adj_aij += v_m[fr]*v_m[to]*(ybus_im.nzval[c]*cos(aij))
        adj_aij *= adj_P
        adj_v_a[to] += -adj_aij
        adj_v_a[fr] += adj_aij
    end
end

function put_reactive_power_injection!(fr, v_m, v_a, adj_v_m, adj_v_a, adj_P, ybus_re, ybus_im)
    @inbounds for c in ybus_re.colptr[fr]:ybus_re.colptr[fr+1]-1
        to = ybus_re.rowval[c]
        aij = v_a[fr] - v_a[to]
        cθ = ybus_im.nzval[c]*cos(aij)
        sθ = ybus_re.nzval[c]*sin(aij)
        adj_v_m[fr] += v_m[to] * (sθ - cθ) * adj_P
        adj_v_m[to] += v_m[fr] * (sθ - cθ) * adj_P

        adj_aij = v_m[fr]*v_m[to]*(ybus_re.nzval[c]*cos(aij))
        adj_aij += v_m[fr]*v_m[to]*(ybus_im.nzval[c]*(sin(aij)))
        adj_aij *= adj_P
        adj_v_a[to] += -adj_aij
        adj_v_a[fr] += adj_aij
    end
end

function put!(
    polar::PolarForm{T, IT, VT, MT},
    ::State,
    ∂x::VT,
    ∂vmag::VT,
    ∂vang::VT,
) where {T, IT, VT, MT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    # build vector x
    for (i, p) in enumerate(polar.network.pv)
        ∂x[i] = ∂vang[p]
    end
    for (i, p) in enumerate(polar.network.pq)
        ∂x[npv+i] = ∂vang[p]
        ∂x[npv+npq+i] = ∂vmag[p]
    end
end

function put!(
    polar::PolarForm{T, IT, VT, MT},
    ::Control,
    ∂u::VT,
    ∂vmag::VT,
    ∂pbus::VT,
) where {T, IT, VT, MT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    # build vector u
    for (i, p) in enumerate(polar.network.ref)
        ∂u[i] = ∂vmag[p]
    end
    for (i, p) in enumerate(polar.network.pv)
        ∂u[nref + i] = ∂vmag[p]
        ∂u[nref + npv + i] = 0.0
    end
end

KA.@kernel function put_adjoint_kernel!(
    adj_u, adj_x, adj_vmag, adj_vang, adj_pg,
    index_pv, index_pq, index_ref, pv_to_gen,
)
    i = @index(Global, Linear)
    npv = length(index_pv)
    npq = length(index_pq)
    nref = length(index_ref)

    # PQ buses
    if i <= npq
        bus = index_pq[i]
        adj_x[npv+i] =  adj_vang[bus]
        adj_x[npv+npq+i] = adj_vmag[bus]
    # PV buses
    elseif i <= npq + npv
        i_ = i - npq
        bus = index_pv[i_]
        i_gen = pv_to_gen[i_]
        adj_u[nref + i_] = adj_vmag[bus]
        adj_u[nref + npv + i_] += adj_pg[i_gen]
        adj_x[i_] = adj_vang[bus]
    # SLACK buses
    elseif i <= npq + npv + nref
        i_ = i - npq - npv
        bus = index_ref[i_]
        adj_u[i_] = adj_vmag[bus]
    end
end

function put(
    polar::PolarForm{T, IT, VT, MT},
    ::PS.Generators,
    ::PS.ActivePower,
    obj_autodiff::AdjointStackObjective,
    buffer::PolarNetworkState
) where {T, IT, VT, MT}
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())

    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    index_pv = polar.indexing.index_pv
    index_ref = polar.indexing.index_ref
    index_pq = polar.indexing.index_pq
    pv_to_gen = polar.indexing.index_pv_to_gen
    ref_to_gen = polar.indexing.index_ref_to_gen

    # Get voltages. This is only needed to get the size of adjvmag and adjvang
    vmag = buffer.vmag
    vang = buffer.vang

    adj_pg = obj_autodiff.∂pg
    adj_x = obj_autodiff.∇fₓ
    adj_u = obj_autodiff.∇fᵤ
    adj_vmag = obj_autodiff.∂vm
    adj_vang = obj_autodiff.∂va
    fill!(adj_vmag, 0.0)
    fill!(adj_vang, 0.0)
    fill!(adj_x, 0.0)
    fill!(adj_u, 0.0)

    for i in 1:nref
        bus = index_ref[i]
        i_gen = ref_to_gen[i]
        # pg[i] = inj + polar.active_load[bus]
        adj_inj = adj_pg[i_gen]
        put_active_power_injection!(bus, vmag, vang, adj_vmag, adj_vang, adj_inj, ybus_re, ybus_im)
    end
    if isa(adj_x, Array)
        kernel! = put_adjoint_kernel!(KA.CPU(), 1)
    else
        kernel! = put_adjoint_kernel!(KA.CUDADevice(), 256)
    end
    ev = kernel!(adj_u, adj_x, adj_vmag, adj_vang, adj_pg,
                 index_pv, index_pq, index_ref, pv_to_gen,
                 ndrange=nbus)
    wait(ev)

    return
end

function ∂cost(polar::PolarForm, ∂obj::AdjointStackObjective, buffer::PolarNetworkState)
    pg = buffer.pg
    coefs = polar.costs_coefficients
    c3 = @view coefs[:, 3]
    c4 = @view coefs[:, 4]
    # Return adjoint of quadratic cost
    ∂obj.∂pg .= c3 .+ 2.0 .* c4 .* pg
    put(polar, PS.Generators(), PS.ActivePower(), ∂obj, buffer)
    return
end

function hessian_cost(polar::PolarForm, ∂obj::AdjointStackObjective, buffer::PolarNetworkState)
    coefs = polar.costs_coefficients
    c3 = @view coefs[:, 3]
    c4 = @view coefs[:, 4]
    # Change ordering from pg to [ref; pv]
    pv2gen = polar.indexing.index_pv_to_gen
    ref2gen = polar.indexing.index_ref_to_gen
    ∂pg = (c3 .+ 2.0 .* c4 .* buffer.pg)[ref2gen]
    ∂²pg = 2.0 .* c4[[ref2gen; pv2gen]]

    # Evaluate Hess-vec of objective function f(x, u)
    ∂₂P = active_power_hessian(polar, buffer)::FullSpaceHessian{SparseMatrixCSC{Float64, Int}}

    ∂Pₓ = active_power_jacobian(polar, State(), buffer)::SparseMatrixCSC{Float64, Int}
    ∂Pᵤ = active_power_jacobian(polar, Control(), buffer)::SparseMatrixCSC{Float64, Int}

    D = Diagonal(∂²pg)
    ∇²fₓₓ = ∂pg .* ∂₂P.xx + ∂Pₓ' * D * ∂Pₓ
    ∇²fᵤᵤ = ∂pg .* ∂₂P.uu + ∂Pᵤ' * D * ∂Pᵤ
    ∇²fₓᵤ = ∂pg .* ∂₂P.xu + ∂Pᵤ' * D * ∂Pₓ

    return FullSpaceHessian(∇²fₓₓ, ∇²fₓᵤ, ∇²fᵤᵤ)
end

