
function matpower_jacobian(polar::PolarForm, X::Union{State, Control}, func::PowerFlowBalance, V)
    nbus = get(polar, PS.NumberOfBuses())
    pf = polar.network
    ref, pv, pq = index_buses_host(polar)
    nref = length(ref)
    npv = length(pv)
    npq = length(pq)
    Ybus = pf.Ybus

    dSbus_dVm, dSbus_dVa = PS.matpower_residual_jacobian(V, Ybus)

    if isa(X, State)
        j11 = real(dSbus_dVa[[pv; pq], [pv; pq]])
        j12 = real(dSbus_dVm[[pv; pq], pq])
        j21 = imag(dSbus_dVa[pq, [pv; pq]])
        j22 = imag(dSbus_dVm[pq, pq])
        return [j11 j12; j21 j22]::SparseMatrixCSC{Float64, Int}
    elseif isa(X, Control)
        j11 = real(dSbus_dVm[[pv; pq], [ref; pv]])
        j12 = sparse(I, npv + npq, npv)
        j21 = imag(dSbus_dVm[pq, [ref; pv]])
        j22 = spzeros(npq, npv)
        return [j11 -j12; j21 j22]::SparseMatrixCSC{Float64, Int}
    end
end

function matpower_jacobian(
    polar::PolarForm,
    X::Union{State,Control},
    func::VoltageMagnitudePQ,
    V,
)

    m = size(func)[1]
    nᵤ = get(polar, NumberOfControl())
    nₓ = get(polar, NumberOfState())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    shift = npq + npv

    I = 1:m
    J = (shift+1):(shift+npq)
    V = ones(m)
    if isa(X, State)
        return sparse(I, J, V, m, nₓ)
    elseif isa(X, Control)
        return spzeros(m, nᵤ)
    end
end

function matpower_jacobian(polar::PolarForm, X::Union{State,Control}, func::LineFlows, V)
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

