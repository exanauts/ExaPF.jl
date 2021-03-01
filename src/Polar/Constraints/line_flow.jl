is_constraint(::typeof(flow_constraints)) = true

# Branch flow constraints
function _flow_constraints(polar::PolarForm, cons, vmag, vang)
    nlines = PS.get(polar.network, PS.NumberOfLines())
    ev = branch_flow_kernel!(polar.device)(
        cons, vmag, vang,
        polar.topology.yff_re, polar.topology.yft_re, polar.topology.ytf_re, polar.topology.ytt_re,
        polar.topology.yff_im, polar.topology.yft_im, polar.topology.ytf_im, polar.topology.ytt_im,
        polar.topology.f_buses, polar.topology.t_buses, nlines,
        ndrange=nlines,
    )
    wait(ev)
    return
end

function flow_constraints(polar::PolarForm, cons, buffer::PolarNetworkState)
    _flow_constraints(polar, cons, buffer.vmag, buffer.vang)
end

# Specialized function for AD with ForwardDiff
function flow_constraints(polar::PolarForm, cons, vm, va, pbus, qbus)
    _flow_constraints(polar, cons, vm, va)
end

function flow_constraints_grad!(polar::PolarForm, cons_grad, buffer, weights)
    nlines = PS.get(polar.network, PS.NumberOfLines())
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    PT = polar.topology
    fill!(cons_grad, 0)
    adj_vmag = @view cons_grad[1:nbus]
    adj_vang = @view cons_grad[nbus+1:2*nbus]
    ev = adj_branch_flow_kernel!(polar.device)(weights, buffer.vmag, adj_vmag,
            buffer.vang, adj_vang,
            PT.yff_re, PT.yft_re, PT.ytf_re, PT.ytt_re,
            PT.yff_im, PT.yft_im, PT.ytf_im, PT.ytt_im,
            PT.f_buses, PT.t_buses, nlines, ndrange = nlines
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

function adjoint!(
    polar::PolarForm,
    ::typeof(flow_constraints),
    cons, ∂cons,
    vm, ∂vm,
    va, ∂va,
    pinj, ∂pinj,
    qinj, ∂qinj,
)
    nlines = PS.get(polar.network, PS.NumberOfLines())
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    top = polar.topology

    ev = adj_branch_flow_kernel!(polar.device)(
        ∂cons,
        vm, ∂vm,
        va, ∂va,
        top.yff_re, top.yft_re, top.ytf_re, top.ytt_re,
        top.yff_im, top.yft_im, top.ytf_im, top.ytt_im,
        top.f_buses, top.t_buses, nlines,
        ndrange = nlines
    )
    wait(ev)
end

function matpower_jacobian(polar::PolarForm, ::State, ::typeof(flow_constraints), V)
    nbus = get(polar, PS.NumberOfBuses())
    pf = polar.network
    ref = pf.ref
    pv = pf.pv
    pq = pf.pq
    lines = pf.lines

    dSl_dVm, dSl_dVa = _matpower_lineflow_power_jacobian(V, lines)
    j11 = dSl_dVa[:, [pv; pq]]
    j12 = dSl_dVm[:, pq]
    return [j11 j12]
end

function matpower_jacobian(polar::PolarForm, ::Control, ::typeof(flow_constraints), V)
    nbus   = get(polar, PS.NumberOfBuses())
    nlines = get(polar, PS.NumberOfLines())
    pf = polar.network
    ref = pf.ref ; nref = length(ref)
    pv  = pf.pv  ; npv  = length(pv)
    pq  = pf.pq  ; npq  = length(pq)
    lines = pf.lines

    dSl_dVm, _ = _matpower_lineflow_power_jacobian(V, lines)
    j11 = dSl_dVm[:, [ref; pv]]
    j12 = spzeros(2 * nlines, npv)
    return [j11 j12]
end

