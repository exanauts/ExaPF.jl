is_constraint(::typeof(flow_constraints)) = true

# Branch flow constraints
function flow_constraints(polar::PolarForm, cons, buffer)
    if isa(cons, CUDA.CuArray)
        kernel! = branch_flow_kernel!(KA.CUDADevice())
    else
        kernel! = branch_flow_kernel!(KA.CPU())
    end
    nlines = PS.get(polar.network, PS.NumberOfLines())
    ev = kernel!(
        cons, buffer.vmag, buffer.vang,
        polar.topology.yff_re, polar.topology.yft_re, polar.topology.ytf_re, polar.topology.ytt_re,
        polar.topology.yff_im, polar.topology.yft_im, polar.topology.ytf_im, polar.topology.ytt_im,
        polar.topology.f_buses, polar.topology.t_buses, nlines,
        ndrange=nlines,
    )
    wait(ev)
    return
end

function flow_constraints_grad!(polar::PolarForm, cons_grad, buffer, weights)
    T = typeof(buffer.vmag)
    if isa(buffer.vmag, Array)
        kernel! = accumulate_view!(KA.CPU())
    else
        kernel! = accumulate_view!(KA.CUDADevice())
    end

    nlines = PS.get(polar.network, PS.NumberOfLines())
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    f = polar.topology.f_buses
    t = polar.topology.t_buses
    # Statements references below (1)
    fr_vmag = buffer.vmag[f]
    to_vmag = buffer.vmag[t]
    fr_vang = buffer.vang[f]
    to_vang = buffer.vang[t]
    function lumping(fr_vmag, to_vmag, fr_vang, to_vang)
        Δθ = fr_vang .- to_vang
        cosθ = cos.(Δθ)
        sinθ = sin.(Δθ)
        cons = branch_flow_kernel_zygote(
            polar.topology.yff_re, polar.topology.yft_re, polar.topology.ytf_re, polar.topology.ytt_re,
            polar.topology.yff_im, polar.topology.yft_im, polar.topology.ytf_im, polar.topology.ytt_im,
            fr_vmag, to_vmag,
            cosθ, sinθ
        )
        return dot(weights, cons)
    end
    grad = Zygote.gradient(lumping, fr_vmag, to_vmag, fr_vang, to_vang)
    gvmag = @view cons_grad[1:nbus]
    gvang = @view cons_grad[nbus+1:2*nbus]
    # This is basically the adjoint of the statements above (1). = becomes +=.
    ev_vmag = kernel!(gvmag, grad[1], f, ndrange = nbus)
    ev_vang = kernel!(gvang, grad[3], f, ndrange = nbus)
    wait(ev_vmag)
    wait(ev_vang)
    ev_vmag = kernel!(gvmag, grad[2], t, ndrange = nbus)
    ev_vang = kernel!(gvang, grad[4], t, ndrange = nbus)
    wait(ev_vmag)
    wait(ev_vang)
    return cons_grad
end

function size_constraint(polar::PolarForm{T, IT, VT, MT}, ::typeof(flow_constraints)) where {T, IT, VT, MT}
    return 2 * PS.get(polar.network, PS.NumberOfLines())
end

function bounds(polar::PolarForm{T, IT, VT, MT}, ::typeof(flow_constraints)) where {T, IT, VT, MT}
    f_min, f_max = PS.bounds(polar.network, PS.Lines(), PS.ActivePower())
    return convert(VT, [f_min; f_min]), convert(VT, [f_max; f_max])
end

# Jacobian-transpose vector product
function jtprod(
    polar::PolarForm,
    ::typeof(flow_constraints),
    ∂jac,
    buffer,
    v::AbstractVector,
)
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    index_pv = polar.indexing.index_pv
    index_pq = polar.indexing.index_pq
    index_ref = polar.indexing.index_ref
    pv_to_gen = polar.indexing.index_pv_to_gen
    # Init buffer
    adj_x = ∂jac.∇fₓ
    adj_u = ∂jac.∇fᵤ
    adj_vmag = ∂jac.∂vm
    adj_vang = ∂jac.∂va
    adj_pg = ∂jac.∂pg
    ∇Jv = ∂jac.∂flow
    fill!(adj_pg, 0.0)
    fill!(adj_x, 0.0)
    fill!(adj_u, 0.0)
    fill!(∇Jv, 0.0)
    # Compute gradient w.r.t. vmag and vang
    flow_constraints_grad!(polar, ∇Jv, buffer, v)
    # Copy results into buffer
    copyto!(adj_vmag, ∇Jv[1:nbus])
    copyto!(adj_vang, ∇Jv[nbus+1:2*nbus])
    if isa(adj_x, Array)
        kernel! = put_adjoint_kernel!(KA.CPU())
    else
        kernel! = put_adjoint_kernel!(KA.CUDADevice())
    end
    ev = kernel!(adj_u, adj_x, adj_vmag, adj_vang, adj_pg,
                 index_pv, index_pq, index_ref, pv_to_gen,
                 ndrange=nbus)
    wait(ev)
end

