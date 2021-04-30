is_constraint(::typeof(active_power_constraints)) = true

# g = [P_ref]
function active_power_constraints(polar::PolarForm, cons, buffer)
    ref_to_gen = polar.indexing.index_ref_to_gen
    # Constraint on P_ref (generator) (P_inj = P_g - P_load)
    # NB: Active power generation has been updated previously inside buffer
    copy!(cons, buffer.pg[ref_to_gen])
    return
end

# Function for AutoDiff
function active_power_constraints(polar::PolarForm, cons, vmag, vang, pinj, qinj, pd, qd)
    kernel! = active_power_kernel!(polar.device)
    ref = polar.indexing.index_ref
    ref_to_gen = polar.indexing.index_ref_to_gen
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    for i_ in 1:length(ref)
        bus = ref[i_]
        i_gen = ref_to_gen[i_]
        inj = 0.0
        @inbounds for c in ybus_re.colptr[bus]:ybus_re.colptr[bus+1]-1
            to = ybus_re.rowval[c]
            aij = vang[bus] - vang[to]
            # f_re = a * cos + b * sin
            # f_im = a * sin - b * cos
            coef_cos = vmag[bus]*vmag[to]*ybus_re.nzval[c]
            coef_sin = vmag[bus]*vmag[to]*ybus_im.nzval[c]
            cos_val = cos(aij)
            sin_val = sin(aij)
            inj += coef_cos * cos_val + coef_sin * sin_val
        end
        cons[i_] = inj + pd[bus]
    end
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
        cosθ = cos(aij)
        sinθ = sin(aij)
        cθ = ybus_re.nzval[c]*cosθ
        sθ = ybus_im.nzval[c]*sinθ
        adj_v_m[fr] += v_m[to] * (cθ + sθ) * adj_P
        adj_v_m[to] += v_m[fr] * (cθ + sθ) * adj_P

        adj_aij = -(v_m[fr]*v_m[to]*(ybus_re.nzval[c]*sinθ))
        adj_aij += v_m[fr]*v_m[to]*(ybus_im.nzval[c]*cosθ)
        adj_aij *= adj_P
        adj_v_a[to] += -adj_aij
        adj_v_a[fr] += adj_aij
    end
end

# Adjoint
function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    pg, ∂pg,
    vm, ∂vm,
    va, ∂va,
    pinj, ∂pinj,
    pload, qload,
) where {F<:typeof(active_power_constraints), S, I}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    index_ref = polar.indexing.index_ref
    ref_to_gen = polar.indexing.index_ref_to_gen

    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())
    # Constraint on P_ref (generator) (P_inj = P_g - P_load)
    for i in 1:nref
        ibus = index_ref[i]
        igen = ref_to_gen[i]
        _put_active_power_injection!(ibus, vm, va, ∂vm, ∂va, ∂pg[i], ybus_re, ybus_im)
    end
    return
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

    dSbus_dVm, dSbus_dVa = PS.matpower_residual_jacobian(V, Ybus)
    # w.r.t. state
    if isa(X, State)
        j11 = real(dSbus_dVa[ref, [pv; pq]])
        j12 = real(dSbus_dVm[ref, pq])
    # w.r.t. control
    elseif isa(X, Control)
        j11 = real(dSbus_dVm[ref, [ref; pv]])
        j12 = spzeros(length(ref), npv)
    end
    return [j11 j12]::SparseMatrixCSC{Float64, Int}
end

function matpower_hessian(polar::PolarForm, ::typeof(active_power_constraints), buffer, λ)
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    ref = polar.indexing.index_ref
    # Check consistency: currently only support a single slack node
    @assert length(λ) == 1
    V = buffer.vmag .* exp.(im .* buffer.vang) |> Array
    hxx, hxu, huu = PS.active_power_hessian(V, polar.network.Ybus, pv, pq, ref)

    λₚ = λ[1]
    return FullSpaceHessian(
        λₚ .* hxx,
        λₚ .* hxu,
        λₚ .* huu,
    )
end

