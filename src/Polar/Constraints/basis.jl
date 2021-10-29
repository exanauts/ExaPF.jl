function network_basis end
is_constraint(::typeof(network_basis)) = true

function size_constraint(polar::PolarForm, ::typeof(network_basis))
    return PS.get(polar.network, PS.NumberOfBuses()) + 2 * PS.get(polar.network, PS.NumberOfLines())
end

# We add constraint only on vmag_pq
function _network_basis(polar::PolarForm, cons, vmag, vang)
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    nlines = PS.get(polar.network, PS.NumberOfLines())

    ev = basis_kernel!(polar.device)(
        cons, vmag, vang,
        polar.topology.f_buses, polar.topology.t_buses, nlines, nbus,
        ndrange=(2 * nlines+nbus, size(cons, 2)),
        dependencies=Event(polar.device)
    )
    wait(ev)
    return
end

function network_basis(polar::PolarForm, cons, vmag, vang, pnet, qnet, pload, qload)
    _network_basis(polar, cons, vmag, vang)
end
function network_basis(polar::PolarForm, cons, buffer)
    _network_basis(polar, cons, buffer.vmag, buffer.vang)
end

function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    cons, ∂cons,
    vmag, ∂vmag,
    vang, ∂vang,
    pnet, ∂pnet,
    pload, qload,
) where {F<:typeof(network_basis), S, I}
    nl = PS.get(polar.network, PS.NumberOfLines())
    nb = PS.get(polar.network, PS.NumberOfBuses())
    top = polar.topology
    f = top.f_buses
    t = top.t_buses

    fill!(pbm.intermediate.∂edge_vm_fr , 0.0)
    fill!(pbm.intermediate.∂edge_vm_to , 0.0)
    fill!(pbm.intermediate.∂edge_va_fr , 0.0)
    fill!(pbm.intermediate.∂edge_va_to , 0.0)
    ndrange = (nl+nb, size(∂cons, 2))
    ev = adj_basis_kernel!(polar.device)(
        ∂cons,
        ∂vmag,
        pbm.intermediate.∂edge_vm_fr,
        pbm.intermediate.∂edge_vm_to,
        pbm.intermediate.∂edge_va_fr,
        pbm.intermediate.∂edge_va_to,
        vmag, vang, f, t, nl, nb,
        ndrange=ndrange, dependencies=Event(polar.device),
    )
    wait(ev)

    Cf = pbm.intermediate.Cf
    Ct = pbm.intermediate.Ct
    mul!(∂vmag, Cf, pbm.intermediate.∂edge_vm_fr, 1.0, 1.0)
    mul!(∂vmag, Ct, pbm.intermediate.∂edge_vm_to, 1.0, 1.0)
    mul!(∂vang, Cf, pbm.intermediate.∂edge_va_fr, 1.0, 1.0)
    mul!(∂vang, Ct, pbm.intermediate.∂edge_va_to, 1.0, 1.0)
    return
end

function matpower_jacobian(polar::PolarForm, X::Union{State, Control}, ::typeof(network_basis), V)
    nbus = get(polar, PS.NumberOfBuses())
    nlines = get(polar, PS.NumberOfLines())
    pf = polar.network
    ref, pv, pq = index_buses_host(polar)
    nref = length(ref)
    npv = length(pv)
    npq = length(pq)

    dS_dVm, dS_dVa = PS._matpower_basis_jacobian(V, pf.lines)
    dV2 = 2 * sparse(1:nbus, 1:nbus, abs.(V), nbus, nbus)

    if isa(X, State)
        j11 = real(dS_dVa[:, [pv; pq]])
        j12 = real(dS_dVm[:, pq])
        j21 = imag(dS_dVa[:, [pv; pq]])
        j22 = imag(dS_dVm[:, pq])
        j31 = spzeros(nbus, npv + npq)
        j32 = dV2[:, pq]
        return [j11 j12; j21 j22; j31 j32]::SparseMatrixCSC{Float64, Int}
    elseif isa(X, Control)
        j11 = real(dS_dVm[:, [ref; pv]])
        j12 = spzeros(nlines, npv)
        j21 = imag(dS_dVm[:, [ref; pv]])
        j22 = spzeros(nlines, npv)
        j31 = dV2[:, [ref; pv]]
        j32 = spzeros(nbus, npv)
        return [j11 j12; j21 j22; j31 j32]::SparseMatrixCSC{Float64, Int}
    end
end
