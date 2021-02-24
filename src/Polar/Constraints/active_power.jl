is_constraint(::typeof(active_power_constraints)) = true

# g = [P_ref]
function active_power_constraints(polar::PolarForm, cons, buffer)
    ref_to_gen = polar.indexing.index_ref_to_gen
    # Constraint on P_ref (generator) (P_inj = P_g - P_load)
    # NB: Active power generation has been updated previously inside buffer
    copy!(cons, buffer.pg[ref_to_gen])
    return
end

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
