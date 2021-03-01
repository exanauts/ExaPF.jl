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

function _put_active_power_injection!(fr, v_m, v_a, adj_v_m, adj_v_a, adj_P, ybus_re, ybus_im)
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

# Adjoint
function adjoint!(
    polar::PolarForm,
    ::typeof(active_power_constraints),
    pg, ∂pg,
    vm, ∂vm,
    va, ∂va,
    pinj, ∂pinj,
    qinj, ∂qinj,
)
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    index_ref = polar.indexing.index_ref
    ref_to_gen = polar.indexing.index_ref_to_gen

    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())
    # Constraint on P_ref (generator) (P_inj = P_g - P_load)
    for i in 1:nref
        ibus = index_ref[i]
        igen = ref_to_gen[i]
        _put_active_power_injection!(ibus, vm, va, ∂vm, ∂va, ∂pg[igen], ybus_re, ybus_im)
    end
    return
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

# MATPOWER Jacobian
function matpower_jacobian(polar::PolarForm, X::Union{State,Control}, ::typeof(active_power_constraints), V)
    nbus = get(polar, PS.NumberOfBuses())
    pf = polar.network
    ref = pf.ref ; nref = length(ref)
    pv = pf.pv ; npv = length(pv)
    pq = pf.pq ; npq = length(pq)
    gen2bus = polar.indexing.index_generators
    ngen = length(gen2bus)
    Ybus = pf.Ybus

    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, Ybus)
    # w.r.t. state
    if isa(X, State)
        j11 = real(dSbus_dVa[ref, [pv; pq]])
        j12 = real(dSbus_dVm[ref, pq])
        return [
            j11 j12 ;
            spzeros(length(pv), length(pv) + 2 * length(pq))
        ]::SparseMatrixCSC{Float64, Int}
    # w.r.t. control
    elseif isa(X, Control)
        j11 = real(dSbus_dVm[ref, [ref; pv]])
        j12 = sparse(I, npv, npv)
        return [
            j11 spzeros(length(ref), npv) ;
            spzeros(npv, ngen) j12
        ]::SparseMatrixCSC{Float64, Int}
    end
end

