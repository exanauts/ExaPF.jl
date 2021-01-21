# Hessian vector-product H*λ, H = Jₓₓ (from Matpower)
# Suppose ordering is correct
function _matpower_hessian(V, Ybus, λ)
    n = size(V, 1)
    # build up auxiliary matrices
    Ibus = Ybus * V
    diagV       = sparse(1:n, 1:n, V, n, n)
    diagIbus    = sparse(1:n, 1:n, Ibus, n, n)
    d_invV      = sparse(1:n, 1:n, 1 ./ abs.(V), n, n)
    d_lambda    = sparse(1:n, 1:n, λ, n, n)

    A = sparse(1:n, 1:n, λ .* V, n, n)
    B = Ybus * diagV
    C = A * conj(B)
    D = Ybus' * diagV
    Dl = sparse(1:n, 1:n, D * λ, n, n)
    E = conj(diagV)  * (D * d_lambda - Dl)
    F = C - A * sparse(1:n, 1:n, conj(Ibus), n, n)
    G = d_invV

    # θθ
    G11 = E + F
    # vθ
    G21 = 1im .* G * (E - F)
    # vv
    G22 = G * (C + transpose(C)) * G
    return (G11, transpose(G21), G22)
end

function residual_hessian(::State, ::State, V, Ybus, λ, pv, pq, ref)
    # decompose vector
    n = length(V)
    λp = zeros(n) ; λq = zeros(n)
    npv = length(pv)
    npq = length(pq)
    λp[pv] = λ[1:npv]
    λp[pq] = λ[npv+1:npv+npq]
    λq[pq] = λ[npv+npq+1:end]

    G11, G12, G22 = _matpower_hessian(V, Ybus, λp)
    Pθθ = real.(G11)
    Pvθ = real.(G12)
    Pvv = real.(G22)

    G11, G12, G22 = _matpower_hessian(V, Ybus, λq)
    Qθθ = imag.(G11)
    Qvθ = imag.(G12)
    Qvv = imag.(G22)

    H11 = Pθθ[pv, pv] + Qθθ[pv, pv]
    H12 = Pθθ[pv, pq] + Qθθ[pv, pq]
    H13 = Pvθ[pv, pq] + Qvθ[pv, pq]
    H22 = Pθθ[pq, pq] + Qθθ[pq, pq]
    H23 = Pvθ[pq, pq] + Qvθ[pq, pq]
    H33 = Pvv[pq, pq] + Qvv[pq, pq]

    H = [
        H11  H12  H13
        H12' H22  H23
        H13' H23' H33
    ]

    return H
end

function residual_hessian(::Control, ::Control, V, Ybus, λ, pv, pq, ref)
    # decompose vector
    n = length(V)
    λp = zeros(n) ; λq = zeros(n)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)
    # λ is ordered according to the state x
    λp[pv] = λ[1:npv]
    λp[pq] = λ[npv+1:npv+npq]
    λq[pq] = λ[npv+npq+1:end]

    _, _, G22 = _matpower_hessian(V, Ybus, λp)
    Pvv = real.(G22)

    _, _, G22 = _matpower_hessian(V, Ybus, λq)
    Qvv = imag.(G22)

    H11 = Pvv[ref, ref] + Qvv[ref, ref]
    H12 = Pvv[ref,  pv] + Qvv[ref,  pv]
    H22 = Pvv[pv,   pv] + Qvv[pv,   pv]

    H = [
         H11  H12 spzeros(nref, npv)
         H12' H22 spzeros(npv, npv)
         spzeros(npv, nref + 2 * npv)
    ]

    return H
end

function residual_hessian(::State, ::Control, V, Ybus, λ, pv, pq, ref)
    # decompose vector
    n = length(V)
    λp = zeros(n) ; λq = zeros(n)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)
    # λ is ordered according to the state x
    λp[pv] = λ[1:npv]
    λp[pq] = λ[npv+1:npv+npq]
    λq[pq] = λ[npv+npq+1:end]

    G11, G12, G22 = _matpower_hessian(V, Ybus, λp)
    Pvθ = real.(transpose(G12))
    Pvv = real.(G22)

    G11, G12, G22 = _matpower_hessian(V, Ybus, λq)
    Qvθ = imag.(transpose(G12))
    Qvv = imag.(G22)

    H11 = Pvθ[ref, pv] + Qvθ[ref, pv]
    H12 = Pvθ[ref, pq] + Qvθ[ref, pq]
    H13 = Pvv[ref, pq] + Qvv[ref, pq]
    H21 = Pvθ[pv,  pv] + Qvθ[pv,  pv]
    H22 = Pvθ[pv,  pq] + Qvθ[pv,  pq]
    H23 = Pvv[pv,  pq] + Qvv[pv,  pq]

    H = [
        H11  H12  H13
        H21  H22  H23
        spzeros(npv, npv + 2 * npq)
    ]

    return H
end

function residual_hessian(
    polar::PolarForm,
    r::AbstractVariable,
    s::AbstractVariable,
    buffer::PolarNetworkState,
    λ::AbstractVector,
)
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    V = buffer.vmag .* exp.(im .* buffer.vang)
    return residual_hessian(r, s, V, polar.network.Ybus, λ, pv, pq, ref)
end


# ∂²pg_ref / ∂²x
function active_power_hessian(::State, ::State, V, Ybus, pv, pq, ref)
    # decompose vector
    n = length(V)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)

    λp = zeros(n)
    # Pick only components wrt ref nodes
    λp[ref] .= 1.0

    G11, G12, G22 = _matpower_hessian(V, Ybus, λp)
    Pθθ = real.(G11)
    Pvθ = real.(G12)
    Pvv = real.(G22)

    H11 = Pθθ[pv, pv]
    H12 = Pθθ[pv, pq]
    H13 = Pvθ[pv, pq]
    H22 = Pθθ[pq, pq]
    H23 = Pvθ[pq, pq]
    H33 = Pvv[pq, pq]

    H = [
        H11  H12  H13
        H12' H22  H23
        H13' H23' H33
    ]
    return H
end

# ∂²pg_ref / ∂²u
function active_power_hessian(::Control, ::Control, V, Ybus, pv, pq, ref)
    # decompose vector
    n = length(V)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)

    λp = zeros(n)
    # Pick only components wrt ref nodes
    λp[ref] .= 1.0

    G11, G12, G22 = _matpower_hessian(V, Ybus, λp)
    Pvv = real.(G22)

    H11 = Pvv[ref, ref]
    H12 = Pvv[ref, pv]
    H22 = Pvv[pv, pv]

    H = [
         H11  H12 spzeros(nref, npv)
         H12' H22 spzeros(npv, npv)
         spzeros(npv, nref + 2 * npv)
    ]

    return H
end

# ∂²pg_ref / ∂x∂u
function active_power_hessian(::State, ::Control, V, Ybus, pv, pq, ref)
    # decompose vector
    n = length(V)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)

    λp = zeros(n)
    # Pick only components wrt ref nodes
    λp[ref] .= 1.0

    G11, G12, G22 = _matpower_hessian(V, Ybus, λp)
    Pθθ = real.(G11)
    Pvθ = real.(transpose(G12))
    Pvv = real.(G22)

    H11 = Pvθ[ref, pv]
    H12 = Pvθ[ref, pq]
    H13 = Pvv[ref, pq]
    H21 = Pvθ[pv, pv]
    H22 = Pvθ[pv, pq]
    H23 = Pvv[pv, pq]

    H = [
        H11  H12  H13
        H21  H22  H23
        spzeros(npv, npv + 2 * npq)
    ]

    return H
end

function active_power_hessian(
    polar::PolarForm,
    r::AbstractVariable,
    s::AbstractVariable,
    buffer::PolarNetworkState,
)
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    V = buffer.vmag .* exp.(im .* buffer.vang)
    return active_power_hessian(r, s, V, polar.network.Ybus, pv, pq, ref)
end

# ∂²qg / ∂²x * λ
function reactive_power_hessian(::State, ::State, V, Ybus, λ, pv, pq, ref, igen)
    n = length(V)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)

    λp = zeros(n)
    # Take only components wrt generators
    λp[igen] .= λ

    G11, G12, G22 = _matpower_hessian(V, Ybus, λp)
    Qθθ = imag.(G11)
    Qvθ = imag.(G12)
    Qvv = imag.(G22)

    H11 = Qθθ[pv, pv]
    H12 = Qθθ[pv, pq]
    H13 = Qvθ[pv, pq]
    H22 = Qθθ[pq, pq]
    H23 = Qvθ[pq, pq]
    H33 = Qvv[pq, pq]

    H = [
        H11  H12  H13
        H12' H22  H23
        H13' H23' H33
    ]
    return H
end

# ∂²qg / ∂²u * λ
function reactive_power_hessian(::Control, ::Control, V, Ybus, λ, pv, pq, ref, igen)
    # decompose vector
    n = length(V)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)

    λp = zeros(n)
    # Take only components wrt generators
    λp[igen] .= λ

    G11, G12, G22 = _matpower_hessian(V, Ybus, λp)
    Qvv = imag.(G22)

    H11 = Qvv[ref, ref]
    H12 = Qvv[ref, pv]
    H22 = Qvv[pv, pv]

    H = [
         H11  H12 spzeros(nref, npv)
         H12' H22 spzeros(npv, npv)
         spzeros(npv, nref + 2* npv)
    ]

    return H
end

# ∂²qg / ∂x∂u * λ
function reactive_power_hessian(::State, ::Control, V, Ybus, λ, pv, pq, ref, igen)
    # decompose vector
    n = length(V)
    npv = length(pv)
    npq = length(pq)
    nref = length(ref)

    λp = zeros(n)
    # Take only components wrt generators
    λp[igen] .= λ

    G11, G12, G22 = _matpower_hessian(V, Ybus, λp)
    Qθθ = imag.(G11)
    Qvθ = imag.(transpose(G12))
    Qvv = imag.(G22)

    H11 = Qvθ[ref, pv]
    H12 = Qvθ[ref, pq]
    H13 = Qvv[ref, pq]
    H21 = Qvθ[pv, pv]
    H22 = Qvθ[pv, pq]
    H23 = Qvv[pv, pq]

    H = [
        H11  H12  H13
        H21  H22  H23
        spzeros(npv, 2*npq+npv)
    ]

    return H
end

function reactive_power_hessian(
    polar::PolarForm,
    r::AbstractVariable,
    s::AbstractVariable,
    buffer::PolarNetworkState,
    λ::AbstractVector,
)
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    V = buffer.vmag .* exp.(im .* buffer.vang)
    return reactive_power_hessian(r, s, V, polar.network.Ybus, λ, pv, pq, ref, igen)
end

