
is_constraint(::typeof(reactive_power_constraints)) = true

# Here, the power constraints are ordered as:
# g = [qg_gen]
function _reactive_power_constraints(
    qg, v_m, v_a, pinj, qinj, qload,
    ybus_re, ybus_im, pv, pq, ref, pv_to_gen, ref_to_gen, nbus
)
    if isa(qg, Array)
        kernel! = reactive_power_kernel!(KA.CPU())
    else
        kernel! = reactive_power_kernel!(KA.CUDADevice())
    end
    range_ = length(pv) + length(ref)
    ev = kernel!(
        qg,
        v_m, v_a, pinj, qinj,
        pv, ref, pv_to_gen, ref_to_gen,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
        ybus_im.nzval, qload,
        ndrange=range_
    )
    wait(ev)
end

function reactive_power_constraints(polar::PolarForm, cons, buffer)
    # Refresh reactive power generation in buffer
    update!(polar, PS.Generators(), PS.ReactivePower(), buffer)
    # Constraint on Q_ref (generator) (Q_inj = Q_g - Q_load)
    copy!(cons, buffer.qg)
    return
end

# Function for AD with ForwardDiff
function reactive_power_constraints(polar::PolarForm, cons, vm, va, pbus, qbus)
    nbus = length(vm)
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    pv_to_gen = polar.indexing.index_pv_to_gen
    ref_to_gen = polar.indexing.index_ref_to_gen
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())
    _reactive_power_constraints(
        cons, vm, va, pbus, qbus, polar.reactive_load,
        ybus_re, ybus_im, pv, pq, ref, pv_to_gen, ref_to_gen, nbus
    )
end

function size_constraint(polar::PolarForm, ::typeof(reactive_power_constraints))
    return PS.get(polar.network, PS.NumberOfGenerators())
end

function bounds(polar::PolarForm{T, IT, VT, MT}, ::typeof(reactive_power_constraints)) where {T, IT, VT, MT}
    q_min, q_max = PS.bounds(polar.network, PS.Generators(), PS.ReactivePower())
    return convert(VT, q_min), convert(VT, q_max)
end

# Adjoint
function adjoint!(
    polar::PolarForm,
    ::typeof(reactive_power_constraints),
    cons, ∂cons,
    vm, ∂vm,
    va, ∂va,
    pinj, ∂pinj,
    qinj, ∂qinj,
)
    nbus = get(polar, PS.NumberOfBuses())
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    pv_to_gen = polar.indexing.index_pv_to_gen
    ref_to_gen = polar.indexing.index_ref_to_gen
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    adj_reactive_power!(
        cons, ∂cons,
        vm, ∂vm,
        va, ∂va,
        ybus_re, ybus_im,
        pinj, ∂pinj,
        qinj, ∂qinj,
        polar.reactive_load,
        pv, pq, ref, pv_to_gen, ref_to_gen, nbus,
    )
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

function matpower_jacobian(polar::PolarForm, ::State, ::typeof(reactive_power_constraints), V)
    nbus = get(polar, PS.NumberOfBuses())
    pf = polar.network
    ref = pf.ref
    pv = pf.pv
    pq = pf.pq
    gen2bus = polar.indexing.index_generators
    Ybus = pf.Ybus

    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, Ybus)
    j11 = imag(dSbus_dVa[gen2bus, [pv; pq]])
    j12 = imag(dSbus_dVm[gen2bus, pq])
    return [j11 j12]
end
function matpower_jacobian(polar::PolarForm, ::Control, ::typeof(reactive_power_constraints), V)
    nbus = get(polar, PS.NumberOfBuses())
    pf = polar.network
    ref = pf.ref
    pv = pf.pv
    pq = pf.pq
    gen2bus = polar.indexing.index_generators
    Ybus = pf.Ybus
    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, Ybus)
    j11 = imag(dSbus_dVm[gen2bus, [ref; pv]])
    return [j11 spzeros(length(gen2bus), length(pv))]
end

