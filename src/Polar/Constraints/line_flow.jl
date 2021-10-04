is_constraint(::typeof(flow_constraints)) = true

# Branch flow constraints
function _flow_constraints(polar::PolarForm, cons, vmag, vang)
    nlines = PS.get(polar.network, PS.NumberOfLines())
    ev = branch_flow_kernel!(polar.device)(
        cons, vmag, vang,
        polar.topology.yff_re, polar.topology.yft_re, polar.topology.ytf_re, polar.topology.ytt_re,
        polar.topology.yff_im, polar.topology.yft_im, polar.topology.ytf_im, polar.topology.ytt_im,
        polar.topology.f_buses, polar.topology.t_buses, nlines,
        ndrange=(nlines, size(cons, 2)),
        dependencies=Event(polar.device)
    )
    wait(ev)
    return
end

function flow_constraints(polar::PolarForm, cons, buffer::PolarNetworkState)
    _flow_constraints(polar, cons, buffer.vmag, buffer.vang)
end

# Specialized function for AD with ForwardDiff
function flow_constraints(polar::PolarForm, cons, vmag, vang, pnet, qnet, pload, qload)
    _flow_constraints(polar, cons, vmag, vang)
end

function flow_constraints_grad!(polar::PolarForm, cons_grad, buffer, weights)
    nlines = PS.get(polar.network, PS.NumberOfLines())
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    PT = polar.topology
    fill!(cons_grad, 0)
    adj_vmag = @view cons_grad[1:nbus]
    adj_vang = @view cons_grad[nbus+1:2*nbus]
    ∂edge_vm_fr = similar(cons_grad, nlines)
    ∂edge_vm_to = similar(cons_grad, nlines)
    ∂edge_va_fr = similar(cons_grad, nlines)
    ∂edge_va_to = similar(cons_grad, nlines)
    fill!(∂edge_vm_fr, 0)
    fill!(∂edge_vm_to, 0)
    fill!(∂edge_va_fr, 0)
    fill!(∂edge_va_to, 0)
    ev = adj_branch_flow_edge_kernel!(polar.device)(
        weights, buffer.vmag, adj_vmag, buffer.vang, adj_vang,
        ∂edge_va_to, ∂edge_va_fr, ∂edge_vm_to, ∂edge_vm_fr,
        PT.yff_re, PT.yft_re, PT.ytf_re, PT.ytt_re,
        PT.yff_im, PT.yft_im, PT.ytf_im, PT.ytt_im,
        PT.f_buses, PT.t_buses, nlines,
        ndrange=(nlines, 1), dependencies=Event(polar.device)
    )
    wait(ev)
    ev = adj_branch_flow_node_kernel!(polar.device)(
        buffer.vmag, adj_vmag, buffer.vang, adj_vang,
        ∂edge_va_to, ∂edge_va_fr, ∂edge_vm_to, ∂edge_vm_fr,
        PT.f_buses, PT.t_buses, nlines,
        ndrange=(nbus, 1), dependencies=Event(polar.device)
    )
    wait(ev)
    return cons_grad
end

function size_constraint(polar::PolarForm{T, IT, VT, MT}, ::typeof(flow_constraints)) where {T, IT, VT, MT}
    return 2 * PS.get(polar.network, PS.NumberOfLines())
end

function bounds(polar::PolarForm{T, IT, VT, MT}, ::typeof(flow_constraints)) where {T, IT, VT, MT}
    f_min, f_max = PS.bounds(polar.network, PS.Lines(), PS.ActivePower())
    return convert(VT, [f_min; f_min]), convert(VT, [f_max; f_max])
end

function _adjoint_line_flow!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory,
    ∂cons, vmag, ∂vmag, vang, ∂vang,
)
    nlines = PS.get(polar.network, PS.NumberOfLines())
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    nbatch  = size(∂vmag, 2)
    top = polar.topology

    fill!(pbm.intermediate.∂edge_vm_fr , 0.0)
    fill!(pbm.intermediate.∂edge_vm_to , 0.0)
    fill!(pbm.intermediate.∂edge_va_fr , 0.0)
    fill!(pbm.intermediate.∂edge_va_to , 0.0)

    ev = adj_branch_flow_edge_kernel!(polar.device)(
        ∂cons, vmag, ∂vmag, vang, ∂vang,
        pbm.intermediate.∂edge_va_to,
        pbm.intermediate.∂edge_va_fr,
        pbm.intermediate.∂edge_vm_to,
        pbm.intermediate.∂edge_vm_fr,
        top.yff_re, top.yft_re, top.ytf_re, top.ytt_re,
        top.yff_im, top.yft_im, top.ytf_im, top.ytt_im,
        top.f_buses, top.t_buses, nlines,
        ndrange=(nlines, nbatch), dependencies=Event(polar.device)
    )
    wait(ev)
    ev = adj_branch_flow_node_kernel!(polar.device)(
        vmag, ∂vmag, vang, ∂vang,
        pbm.intermediate.∂edge_va_to,
        pbm.intermediate.∂edge_va_fr,
        pbm.intermediate.∂edge_vm_to,
        pbm.intermediate.∂edge_vm_fr,
        top.f_buses, top.t_buses, nlines,
        ndrange=(nbus, nbatch), dependencies=Event(polar.device)
    )
    wait(ev)
end

function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    cons, ∂cons,
    vmag, ∂vmag,
    vang, ∂vang,
    pnet, ∂pnet,
    pload, qload,
) where {F<:typeof(flow_constraints), S, I}
    _adjoint_line_flow!(polar, pbm, ∂cons, vmag, ∂vmag, vang, ∂vang)
end

function matpower_jacobian(polar::PolarForm, X::Union{State,Control}, ::typeof(flow_constraints), V)
    nbus = get(polar, PS.NumberOfBuses())
    nlines = get(polar, PS.NumberOfLines())
    pf = polar.network
    ref, pv, pq = index_buses_host(polar)
    nref = length(ref)
    npv  = length(pv)
    npq  = length(pq)
    lines = pf.lines

    dSl_dVm, dSl_dVa = PS.matpower_lineflow_power_jacobian(V, lines)

    if isa(X, State)
        j11 = dSl_dVa[:, [pv; pq]]
        j12 = dSl_dVm[:, pq]
        return [j11 j12]::SparseMatrixCSC{Float64, Int}
    elseif isa(X, Control)
        j11 = dSl_dVm[:, [ref; pv]]
        j12 = spzeros(2 * nlines, npv)
        return [j11 j12]::SparseMatrixCSC{Float64, Int}
    end
end

