# Matpower Jacobian (order of columns: [vmag, vang, pgen])

function matpower_jacobian(polar::PolarForm, func::PowerFlowBalance, V)
    pf = polar.network
    nbus = pf.nbus
    ngen = pf.ngen
    ref, pv, pq = index_buses_host(polar)
    gen2bus = pf.gen2bus
    nref = length(ref)
    npv = length(pv)
    npq = length(pq)
    Ybus = pf.Ybus

    dSbus_dVm, dSbus_dVa = PS.matpower_residual_jacobian(V, Ybus)

    Cg_tot = sparse(gen2bus, 1:ngen, -ones(ngen), nbus, ngen)
    Cg = Cg_tot[[pv; pq], :]

    j11 = real(dSbus_dVm[[pv; pq], :])
    j12 = real(dSbus_dVa[[pv; pq], :])
    j13 = Cg #sparse(gen2bus, 1:ngen, -ones(ngen), npv + npq, ngen)
    j21 = imag(dSbus_dVm[pq, :])
    j22 = imag(dSbus_dVa[pq, :])
    j23 = spzeros(npq, ngen)

    return [
        j11 j12 j13;
        j21 j22 j23
    ]::SparseMatrixCSC{Float64, Int}
end

function matpower_jacobian(polar::PolarForm, func::VoltageMagnitudePQ, V)
    pf = polar.network
    ngen = pf.ngen
    nbus = pf.nbus
    ref, pv, pq = index_buses_host(polar)
    j11 = sparse(1:npq, pq, ones(npq), npq, nbus)
    j12 = spzeros(npq, nbus + ngen)
    return [j11 j12]::SparseMatrixCSC{Float64, Int}
end

function matpower_jacobian(polar::PolarForm, func::PowerGenerationBounds, V)
    pf = polar.network
    nbus = pf.nbus
    ngen = pf.ngen
    gen2bus = pf.gen2bus
    ref, pv, pq = index_buses_host(polar)
    nref = length(ref)
    npv = length(pv)
    npq = length(pq)
    Ybus = pf.Ybus

    dSbus_dVm, dSbus_dVa = PS.matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVm[ref, :])
    j12 = real(dSbus_dVa[ref, :])
    j13 = spzeros(nref, ngen)

    j21 = imag(dSbus_dVm[gen2bus, :])
    j22 = imag(dSbus_dVa[gen2bus, :])
    j23 = spzeros(ngen, ngen)
    # w.r.t. control
    return [
        j11 j12 j13 ;
        j21 j22 j23
    ]::SparseMatrixCSC{Float64, Int}
end

function matpower_jacobian(polar::PolarForm, func::LineFlows, V)
    nbus = get(polar, PS.NumberOfBuses())
    nlines = get(polar, PS.NumberOfLines())
    pf = polar.network
    ref, pv, pq = index_buses_host(polar)
    ngen = pf.ngen
    nref = length(ref)
    npv  = length(pv)
    npq  = length(pq)
    lines = pf.lines

    dSl_dVm, dSl_dVa = PS.matpower_lineflow_power_jacobian(V, lines)

    j11 = dSl_dVm
    j12 = dSl_dVa
    j13 = spzeros(2 * nlines, ngen)

    return [
        j11 j12 j13;
    ]::SparseMatrixCSC{Float64, Int}
end

