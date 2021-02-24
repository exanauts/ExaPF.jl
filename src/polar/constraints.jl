
# By default, generic Julia functions are not considered as constraint:
is_constraint(::Function) = false

# Generic inequality constraints
# We add constraint only on vmag_pq
function voltage_magnitude_constraints(polar::PolarForm, g, buffer)
    index_pq = polar.indexing.index_pq
    g .= @view buffer.vmag[index_pq]
    return
end

is_constraint(::typeof(voltage_magnitude_constraints)) = true

size_constraint(polar::PolarForm{T, IT, VT, MT}, ::typeof(voltage_magnitude_constraints)) where {T, IT, VT, MT} = PS.get(polar.network, PS.NumberOfPQBuses())

function bounds(polar::PolarForm, ::typeof(voltage_magnitude_constraints))
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    fr_ = npq + npv + 1
    to_ = 2*npq + npv
    return polar.x_min[fr_:to_], polar.x_max[fr_:to_]
end

# State Jacobian: Jx_i = [0, ..., 1, ... 0] where
function jacobian(polar::PolarForm, ::typeof(voltage_magnitude_constraints), i_cons::Int, ∂jac, buffer)
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    fr_ = npq + npv

    # Adjoint / State
    fill!(∂jac.∇fₓ, 0)
    ∂jac.∇fₓ[fr_ + i_cons] = 1.0
    # Adjoint / Control
    fill!(∂jac.∇fᵤ, 0)
end

function jacobian(polar::PolarForm, cons::typeof(voltage_magnitude_constraints), buffer)
    m = size_constraint(polar, cons)
    nᵤ = get(polar, NumberOfControl())
    nₓ = get(polar, NumberOfState())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    shift = npq + npv

    I = 1:m
    J = (shift+1):(shift+npq)
    V = ones(m)
    jx = sparse(I, J, V, m, nₓ)
    ju = spzeros(m, nᵤ)
    return FullSpaceJacobian(jx, ju)
end

function jtprod(polar::PolarForm, ::typeof(voltage_magnitude_constraints), ∂jac, buffer, v)
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    fr_ = npq + npv + 1
    # Adjoint / Control
    fill!(∂jac.∇fᵤ, 0)
    # Adjoint / State
    fill!(∂jac.∇fₓ, 0)
    ∂jac.∇fₓ[fr_:end] .= v
end

function hessian(polar::PolarForm, ::typeof(voltage_magnitude_constraints), buffer, λ)
    nu = get(polar, NumberOfControl())
    nx = get(polar, NumberOfState())
    return FullSpaceHessian(
        spzeros(nx, nx),
        spzeros(nu, nx),
        spzeros(nu, nu),
    )
end

# Here, the power constraints are ordered as:
# g = [qg_gen]
function reactive_power_constraints(polar::PolarForm, cons, buffer)
    # Refresh reactive power generation in buffer
    update!(polar, PS.Generators(), PS.ReactivePower(), buffer)
    # Constraint on Q_ref (generator) (Q_inj = Q_g - Q_load)
    copy!(cons, buffer.qg)
    return
end

is_constraint(::typeof(reactive_power_constraints)) = true

function size_constraint(polar::PolarForm{T, IT, VT, MT}, ::typeof(reactive_power_constraints)) where {T, IT, VT, MT}
    return PS.get(polar.network, PS.NumberOfGenerators())
end

function bounds(polar::PolarForm{T, IT, VT, MT}, ::typeof(reactive_power_constraints)) where {T, IT, VT, MT}
    q_min, q_max = PS.bounds(polar.network, PS.Generators(), PS.ReactivePower())
    return convert(VT, q_min), convert(VT, q_max)
end

# Jacobian
function jacobian(
    polar::PolarForm,
    ::typeof(reactive_power_constraints),
    i_cons,
    ∂jac,
    buffer
)
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    gen2bus = polar.indexing.index_generators
    index_pv = polar.indexing.index_pv
    index_pq = polar.indexing.index_pq
    index_ref = polar.indexing.index_ref
    pv_to_gen = polar.indexing.index_pv_to_gen

    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    vmag = buffer.vmag
    vang = buffer.vang
    adj_x = ∂jac.∇fₓ
    adj_u = ∂jac.∇fᵤ
    adj_vmag = ∂jac.∂vm
    adj_vang = ∂jac.∂va
    adj_pg = ∂jac.∂pg
    fill!(adj_pg, 0.0)
    fill!(adj_vmag, 0.0)
    fill!(adj_vang, 0.0)
    fill!(adj_x, 0.0)
    fill!(adj_u, 0.0)

    adj_inj = 1.0
    bus = gen2bus[i_cons]
    put_reactive_power_injection!(bus, vmag, vang, adj_vmag, adj_vang, adj_inj, ybus_re, ybus_im)
    ev = put_adjoint_kernel!(polar.device)(
        adj_u, adj_x, adj_vmag, adj_vang, adj_pg,
        index_pv, index_pq, index_ref, pv_to_gen,
        ndrange=nbus
    )
    wait(ev)
end

function jacobian(polar::PolarForm, cons::typeof(reactive_power_constraints), buffer)
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    gen2bus = polar.indexing.index_generators
    ngen = length(gen2bus)
    # Use MATPOWER to derive expression of Hessian
    # Use the fact that q_g = q_inj + q_load
    V = buffer.vmag .* exp.(im .* buffer.vang)
    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, polar.network.Ybus)

    # wrt Qg
    Q21x = imag(dSbus_dVa[gen2bus, [pv; pq]])
    Q22x = imag(dSbus_dVm[gen2bus, pq])

    Q21u = imag(dSbus_dVm[gen2bus, ref])
    Q22u = imag(dSbus_dVm[gen2bus, pv])
    Q23u = spzeros(ngen, length(pv))

    jx = [Q21x Q22x]
    ju = [Q21u Q22u Q23u]

    return FullSpaceJacobian(jx, ju)
end

# Jacobian-transpose vector product
function jtprod(
    polar::PolarForm,
    ::typeof(reactive_power_constraints),
    ∂jac,
    buffer,
    v::AbstractVector,
)
    m = size_constraint(polar, reactive_power_constraints)
    jvx = similar(∂jac.∇fₓ) ; fill!(jvx, 0)
    jvu = similar(∂jac.∇fᵤ) ; fill!(jvu, 0)
    for i_cons in 1:m
        if !iszero(v[i_cons])
            jacobian(polar, reactive_power_constraints, i_cons, ∂jac, buffer)
            jx, ju = ∂jac.∇fₓ, ∂jac.∇fᵤ
            jvx .+= jx .* v[i_cons]
            jvu .+= ju .* v[i_cons]
        end
    end
    ∂jac.∇fₓ .= jvx
    ∂jac.∇fᵤ .= jvu
end

function hessian(polar::PolarForm, ::typeof(reactive_power_constraints), buffer, λ)
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    gen2bus = polar.indexing.index_generators
    # Check consistency
    @assert length(λ) == length(gen2bus)

    nu = get(polar, NumberOfControl())
    nx = get(polar, NumberOfState())

    V = buffer.vmag .* exp.(im .* buffer.vang)
    Ybus = polar.network.Ybus

    λq = zeros(length(V))
    # Select only buses with generators
    λq[gen2bus] .= λ
    ∂₂Q = reactive_power_hessian(V, Ybus, λq, pv, pq, ref)
    return FullSpaceHessian(
        ∂₂Q.xx, ∂₂Q.xu, ∂₂Q.uu
    )
end


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

is_constraint(::typeof(flow_constraints)) = true

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

# g = [P_ref; Q_ref; Q_pv]
function active_power_constraints(polar::PolarForm, cons, buffer)
    ref_to_gen = polar.indexing.index_ref_to_gen
    # Constraint on P_ref (generator) (P_inj = P_g - P_load)
    # NB: Active power generation has been updated previously inside buffer
    copy!(cons, buffer.pg[ref_to_gen])
    return
end

is_constraint(::typeof(active_power_constraints)) = true

function size_constraint(polar::PolarForm{T, IT, VT, MT}, ::typeof(active_power_constraints)) where {T, IT, VT, MT}
    return PS.get(polar.network, PS.NumberOfSlackBuses())
end

function bounds(polar::PolarForm{T, IT, VT, MT}, ::typeof(active_power_constraints)) where {T, IT, VT, MT}
    # Get all bounds (lengths of p_min, p_max, q_min, q_max equal to ngen)
    p_min, p_max = PS.bounds(polar.network, PS.Generators(), PS.ActivePower())
    ref_to_gen = polar.indexing.index_ref_to_gen
    pq_min = p_min[ref_to_gen]
    pq_max = p_max[ref_to_gen]
    return convert(VT, pq_min), convert(VT, pq_max)
end

# Jacobian
function jacobian(
    polar::PolarForm,
    ::typeof(active_power_constraints),
    i_cons,
    ∂jac,
    buffer
)
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    index_pv = polar.indexing.index_pv
    index_pq = polar.indexing.index_pq
    index_ref = polar.indexing.index_ref
    pv_to_gen = polar.indexing.index_pv_to_gen

    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    vmag = buffer.vmag
    vang = buffer.vang
    adj_x = ∂jac.∇fₓ
    adj_u = ∂jac.∇fᵤ
    adj_vmag = ∂jac.∂vm
    adj_vang = ∂jac.∂va
    adj_pg = ∂jac.∂pg
    fill!(adj_pg, 0.0)
    fill!(adj_vmag, 0.0)
    fill!(adj_vang, 0.0)
    fill!(adj_x, 0.0)
    fill!(adj_u, 0.0)

    adj_inj = 1.0
    # Constraint on P_ref (generator) (P_inj = P_g - P_load)
    bus = index_ref[i_cons]
    put_active_power_injection!(bus, vmag, vang, adj_vmag, adj_vang, adj_inj, ybus_re, ybus_im)
    ev = put_adjoint_kernel!(polar.device)(adj_u, adj_x, adj_vmag, adj_vang, adj_pg,
                 index_pv, index_pq, index_ref, pv_to_gen,
                 ndrange=nbus)
    wait(ev)
end

function jacobian(polar::PolarForm, cons::typeof(active_power_constraints), buffer)
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    gen2bus = polar.indexing.index_generators
    ngen = length(gen2bus)
    npv = length(pv)
    nref = length(ref)
    # Use MATPOWER to derive expression of Hessian
    # Use the fact that q_g = q_inj + q_load
    V = buffer.vmag .* exp.(im .* buffer.vang)
    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, polar.network.Ybus)

    # wrt Pg_ref
    P11x = real(dSbus_dVa[ref, [pv; pq]])
    P12x = real(dSbus_dVm[ref, pq])
    P11u = real(dSbus_dVm[ref, [ref; pv]])
    P12u = spzeros(nref, npv)

    jx = [P11x P12x]
    ju = [P11u P12u]
    return FullSpaceJacobian(jx, ju)
end

# Jacobian-transpose vector product
function jtprod(
    polar::PolarForm,
    ::typeof(active_power_constraints),
    ∂jac,
    buffer,
    v::AbstractVector,
)
    m = size_constraint(polar, active_power_constraints)
    jvx = similar(∂jac.∇fₓ) ; fill!(jvx, 0)
    jvu = similar(∂jac.∇fᵤ) ; fill!(jvu, 0)
    for i_cons in 1:m
        if !iszero(v[i_cons])
            jacobian(polar, active_power_constraints, i_cons, ∂jac, buffer)
            jx, ju = ∂jac.∇fₓ, ∂jac.∇fᵤ
            jvx .+= jx .* v[i_cons]
            jvu .+= ju .* v[i_cons]
        end
    end
    ∂jac.∇fₓ .= jvx
    ∂jac.∇fᵤ .= jvu
end

function hessian(polar::PolarForm, ::typeof(active_power_constraints), buffer, λ)
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    # Check consistency
    @assert length(λ) == 1

    V = buffer.vmag .* exp.(im .* buffer.vang)
    Ybus = polar.network.Ybus

    # First constraint is on active power generation at slack node
    λₚ = λ[1]
    ∂₂P = active_power_hessian(V, Ybus, pv, pq, ref)

    return FullSpaceHessian(
        λₚ .* ∂₂P.xx,
        λₚ .* ∂₂P.xu,
        λₚ .* ∂₂P.uu,
    )
end

