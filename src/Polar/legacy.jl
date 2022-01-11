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
    npq = length(pq)

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

function matpower_jacobian(polar::PolarForm, func::PolarBasis, V)
    pf = polar.network
    nbus = pf.nbus
    ngen = pf.ngen
    nlines = get(polar, PS.NumberOfLines())

    dS_dVm, dS_dVa = PS._matpower_basis_jacobian(V, pf.lines)
    dV2 = 2 * sparse(1:nbus, 1:nbus, abs.(V), nbus, nbus)

    j11 = real(dS_dVm)
    j12 = real(dS_dVa)
    j13 = spzeros(nlines, ngen)
    j21 = imag(dS_dVm)
    j22 = imag(dS_dVa)
    j23 = spzeros(nlines, ngen)
    j31 = dV2
    j32 = spzeros(nbus, nbus)
    j33 = spzeros(nbus, ngen)
    return [
        j11 j12 j13;
        j21 j22 j23;
        j31 j32 j33
    ]::SparseMatrixCSC{Float64, Int}
end

function matpower_jacobian(polar::PolarForm, func::MultiExpressions, V)
    return vcat([matpower_jacobian(polar, expr, V) for expr in func.exprs]...)
end
matpower_jacobian(polar::PolarForm, func::ComposedExpressions, V) = matpower_jacobian(polar, func.outer, V)

# Return full-space Hessian
function matpower_hessian(polar::PolarForm, func::CostFunction, V, λ)
    pf = polar.network
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    ref = pf.ref
    nref = length(ref)
    c2 = func.c2 |> Array

    H11 = spzeros(2* nbus, 2 * nbus + ngen)

    H21 = spzeros(ngen, 2 * nbus)
    H22 = spdiagm(2.0 .* c2)
    H = [
        H11;
        H21 H22;
    ]::SparseMatrixCSC{Float64, Int}

    # pg_ref is implicit: add term corresponding to ∂pg_ref' * ∂pg_ref
    Ybus = pf.Ybus
    dSbus_dVm, dSbus_dVa = PS.matpower_residual_jacobian(V, Ybus)
    j11 = real(dSbus_dVm[ref, :])
    j12 = real(dSbus_dVa[ref, :])
    j13 = spzeros(nref, ngen)
    J = [j11 j12 j13]::SparseMatrixCSC{Float64, Int}

    Href = J' * Diagonal(2 .* c2[ref]) * J

    return H + Href
end

function matpower_hessian(polar::PolarForm, func::PowerFlowBalance, V, λ)
    pf = polar.network
    Ybus = pf.Ybus
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    pq, pv = pf.pq, pf.pv
    npq, npv = length(pq), length(pv)

    yp = zeros(nbus)
    yp[pv] .= λ[1:npv]
    yp[pq] .= λ[1+npv:npv+npq]
    Hpθθ, Hpvθ, Hpvv = PS._matpower_hessian(V, Ybus, yp)

    yq = zeros(nbus)
    yq[pq] .= λ[1+npv+npq:npv+2*npq]
    Hqθθ, Hqvθ, Hqvv = PS._matpower_hessian(V, Ybus, yq)

    H11 = real.(Hpvv) .+ imag.(Hqvv)
    H12 = real.(Hpvθ) .+ imag.(Hqvθ)
    H13 = spzeros(nbus, ngen)

    H21 = real.(Hpvθ') .+ imag.(Hqvθ')
    H22 = real.(Hpθθ) .+ imag.(Hqθθ)
    H23 = spzeros(nbus, ngen)

    H31 = spzeros(ngen, nbus)
    H32 = spzeros(ngen, nbus)
    H33 = spzeros(ngen, ngen)
    return [
        H11 H12 H13;
        H21 H22 H23;
        H31 H32 H33
    ]::SparseMatrixCSC{Float64, Int}
end

function matpower_hessian(polar::PolarForm, func::PowerGenerationBounds, V, λ)
    pf = polar.network
    Ybus = pf.Ybus
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    ref, pv = pf.ref, pf.pv
    nref, npv = length(ref), length(pv)

    yp = zeros(nbus)
    yp[ref] .= λ[1:nref]
    Hpθθ, Hpvθ, Hpvv = PS._matpower_hessian(V, Ybus, yp)

    yq = zeros(nbus)
    yq[pv] .= λ[nref+1:nref+npv]
    Hqθθ, Hqvθ, Hqvv = PS._matpower_hessian(V, Ybus, yq)

    H11 = real.(Hpvv) .+ imag.(Hqvv)
    H12 = real.(Hpvθ) .+ imag.(Hqvθ)
    H13 = spzeros(nbus, ngen)

    H21 = real.(Hpvθ') .+ imag.(Hqvθ')
    H22 = real.(Hpθθ) .+ imag.(Hqθθ)
    H23 = spzeros(nbus, ngen)

    H31 = spzeros(ngen, nbus)
    H32 = spzeros(ngen, nbus)
    H33 = spzeros(ngen, ngen)
    return [
        H11 H12 H13;
        H21 H22 H23;
        H31 H32 H33
    ]::SparseMatrixCSC{Float64, Int}
end

function matpower_hessian(polar::PolarForm, func::VoltageMagnitudePQ, V, λ)
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    n = 2*nbus + ngen
    return spzeros(n, n)
end

# TODO: not implemented yet
function matpower_hessian(polar::PolarForm, func::LineFlows, V, λ)
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    n = 2*nbus + ngen
    return spzeros(n, n)
end

function matpower_hessian(polar::PolarForm, func::MultiExpressions, V, λ)
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    n = 2*nbus + ngen
    H = spzeros(n, n)

    k = 0
    for expr in func.exprs
        m = length(expr)
        y = view(λ, k+1:k+m)
        H += matpower_hessian(polar, expr, V, y)::SparseMatrixCSC{Float64, Int}
        k += m
    end
    return H
end
matpower_hessian(polar::PolarForm, func::ComposedExpressions, V, y) = matpower_hessian(polar, func.outer, V, y)

function hessian_sparsity(polar::PolarForm, func)
    m = length(func)
    nbus = get(polar, PS.NumberOfBuses())
    Vre = Float64[i for i in 1:nbus]
    Vim = Float64[i for i in nbus+1:2*nbus]
    V = Vre .+ im .* Vim
    y = rand(m)
    return matpower_hessian(polar, func, V, y)
end

