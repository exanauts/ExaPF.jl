is_constraint(::typeof(active_power_constraints)) = true

# g = [P_ref]
function active_power_constraints(polar::PolarForm, cons, buffer)
    ref_to_gen = polar.indexing.index_ref_to_gen
    # Constraint on P_ref (generator) (P_inj = P_g - P_load)
    # NB: Active power generation has been updated previously inside buffer
    copyto!(cons, buffer.pgen[ref_to_gen])
    return
end

# Function for AutoDiff
function active_power_constraints(polar::PolarForm, cons, vmag, vang, pnet, qnet, pd, qd)
    kernel! = active_power_kernel!(polar.device)
    ref = polar.indexing.index_ref
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())
    ndrange = length(ref)
    ev = active_power_slack!(polar.device)(cons, vmag, vang, ref, pd,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval, ybus_im.nzval,
        ndrange=ndrange,
    )
    wait(ev)
end

function size_constraint(polar::PolarForm{T, IT, VT, MT}, ::typeof(active_power_constraints)) where {T, IT, VT, MT}
    return PS.get(polar.network, PS.NumberOfSlackBuses())
end

function bounds(polar::PolarForm{T, IT, VT, MT}, ::typeof(active_power_constraints)) where {T, IT, VT, MT}
    # Get all bounds (lengths of p_min, p_max, q_min, q_max equal to ngen)
    p_min, p_max = PS.bounds(polar.network, PS.Generators(), PS.ActivePower())
    ref_to_gen = polar.indexing.index_ref_to_gen |> Array
    pq_min = p_min[ref_to_gen]
    pq_max = p_max[ref_to_gen]
    return convert(VT, pq_min), convert(VT, pq_max)
end

# Adjoint
function adjoint!(
    polar::PolarForm,
    pbm::AutoDiff.TapeMemory{F, S, I},
    pg, ∂pg,
    vm, ∂vm,
    va, ∂va,
    pnet, ∂pnet,
    pload, qload,
) where {F<:typeof(active_power_constraints), S, I}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    ref = polar.indexing.index_ref

    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())
    ndrange = nref
    ev = adj_active_power_slack!(polar.device)(vm, va, ∂vm, ∂va, ∂pg, ref,
        ybus_re.nzval, ybus_re.colptr, ybus_re.rowval, ybus_im.nzval,
        ndrange=ndrange,
    )
    wait(ev)
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
    pv = polar.network.pv
    pq = polar.network.pq
    ref = polar.network.ref
    # Check consistency: currently only support a single slack node
    @assert length(λ) == 1
    V = buffer.vmag .* exp.(im .* buffer.vang) |> Array
    hxx, hxu, huu = PS.active_power_hessian(V, polar.network.Ybus, pv, pq, ref)

    λₚ = sum(λ)  # TODO
    return FullSpaceHessian(
        λₚ .* hxx,
        λₚ .* hxu,
        λₚ .* huu,
    )
end

