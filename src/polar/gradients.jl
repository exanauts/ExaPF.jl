function _matpower_residual_jacobian(V, Ybus)
    n = size(V, 1)
    Ibus = Ybus*V
    diagV       = sparse(1:n, 1:n, V, n, n)
    diagIbus    = sparse(1:n, 1:n, Ibus, n, n)
    diagVnorm   = sparse(1:n, 1:n, V./abs.(V), n, n)

    dSbus_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
    dSbus_dVa = 1im * diagV * conj(diagIbus - Ybus * diagV)
    return (dSbus_dVm, dSbus_dVa)
end

# Jacobian Jₓ (from Matpower)
"""
    residual_jacobian(V, Ybus, pv, pq)

Compute the Jacobian w.r.t. the state `x` of the power
balance function [`power_balance`](@ref).

# Note
Code adapted from MATPOWER.
"""
function residual_jacobian(::State, V, Ybus, pv, pq, ref)
    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVa[[pv; pq], [pv; pq]])
    j12 = real(dSbus_dVm[[pv; pq], pq])
    j21 = imag(dSbus_dVa[pq, [pv; pq]])
    j22 = imag(dSbus_dVm[pq, pq])

    J = [j11 j12; j21 j22]
end

# Jacobian Jᵤ (from Matpower)
function residual_jacobian(::Control, V, Ybus, pv, pq, ref)
    dSbus_dVm, _ = _matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVm[[pv; pq], [ref; pv; pv]])
    j21 = imag(dSbus_dVm[pq, [ref; pv; pv]])
    J = [j11; j21]
end

function residual_jacobian_bus(::State, bustype, idx_bus_pf, Y)

    nnz = 0
    for fr in 1:size(bustype, 1)
        fr_ty = bustype[fr]
        for c in Y.colptr[fr]:Y.colptr[fr + 1] - 1
            to = Y.rowval[c]
            to_ty = bustype[to]
            if fr_ty == 2 && to_ty == 2
                nnz += 1
            elseif fr_ty == 2 && to_ty == 1
                nnz += 2
            elseif fr_ty == 1 && to_ty == 2
                nnz += 2
            elseif fr_ty == 1 && to_ty == 1
                nnz += 4
            end
        end
    end
    I = zeros(nnz) #row
    J = zeros(nnz) #column
    V = zeros(nnz) #vals
    
    pos = 1
    for fr in 1:size(bustype, 1)
        fr_ty = bustype[fr]
        fr_ptr = idx_bus_pf[fr]
        for c in Y.colptr[fr]:Y.colptr[fr + 1] - 1
            to = Y.rowval[c]
            to_ty = bustype[to]
            to_ptr = idx_bus_pf[to]
            if fr_ty == 2 && to_ty == 2
                I[pos] = fr_ptr
                J[pos] = to_ptr
                pos += 1
            elseif fr_ty == 2 && to_ty == 1
                I[pos] = fr_ptr
                J[pos] = to_ptr
                I[pos + 1] = fr_ptr
                J[pos + 1] = to_ptr + 1
                pos += 2
            elseif fr_ty == 1 && to_ty == 2
                I[pos] = fr_ptr
                J[pos] = to_ptr
                I[pos + 1] = fr_ptr + 1
                J[pos + 1] = to_ptr
                pos += 2
            elseif fr_ty == 1 && to_ty == 1
                I[pos] = fr_ptr
                J[pos] = to_ptr
                I[pos + 1] = fr_ptr + 1
                J[pos + 1] = to_ptr
                I[pos + 2] = fr_ptr
                J[pos + 2] = to_ptr + 1
                I[pos + 3] = fr_ptr + 1
                J[pos + 3] = to_ptr + 1
                pos += 4
            end
        end
    end

    return sparse(I, J, V)

end

function residual_jacobian(A::Attr, polar::PolarForm) where {Attr <: Union{State, Control}}
    pf = polar.network
    ref = polar.network.ref
    pv = polar.network.pv
    pq = polar.network.pq
    n = PS.get(pf, PS.NumberOfBuses())
    bustype = PS.get(pf, PS.BusTypeIndex())
    idx_bus_pf = PS.get(pf, PS.BusPFIndex())

    Y = pf.Ybus
    # Randomized inputs
    Vre = rand(n)
    Vim = rand(n)
    V = Vre .+ 1im .* Vim

    return residual_jacobian_bus(A, bustype, idx_bus_pf, Y)
    #return residual_jacobian(A, V, Y, pv, pq, ref)
end
_sparsity_pattern(polar::PolarForm) = findnz(residual_jacobian(State(), polar))


# Jacobian wrt active power generation
function active_power_jacobian(::State, V, Ybus, pv, pq, ref)
    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVa[:, [pv; pq]])
    j12 = real(dSbus_dVm[:, pq])
    J = [j11 j12]
end
function active_power_jacobian(::Control, V, Ybus, pv, pq, ref)
    dSbus_dVm, dSbus_dVa = _matpower_residual_jacobian(V, Ybus)
    J = real(dSbus_dVm[:, [ref; pv]])
end
