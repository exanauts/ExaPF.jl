
struct PolarFormRecourse{T, IT, VT, MT} <: AbstractPolarFormulation{T, IT, VT, MT}
    network::PS.PowerNetwork
    device::KA.Device
    k::Int
    ncustoms::Int  # custom variables defined by user
end

function PolarFormRecourse(pf::PS.PowerNetwork, device::KA.CPU, k::Int)
    ngen = PS.get(pf, PS.NumberOfGenerators())
    ncustoms = (ngen + 1) * k
    return PolarFormRecourse{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}(pf, device, k, ncustoms)
end
# Convenient constructor
PolarFormRecourse(datafile::String, k::Int, device=CPU()) = PolarFormRecourse(PS.PowerNetwork(datafile), device, k)

name(polar::PolarFormRecourse) = "Polar formulation with recourse"
nblocks(polar::PolarFormRecourse) = polar.k

"""
    ExtendedPowerFlowBalance
"""
struct PowerFlowRecourse{VT, MT} <: AutoDiff.AbstractExpression
    M::MT
    Cg::MT
    Cdp::MT
    Cdq::MT
    pgmin::VT
    pgmax::VT
    alpha::VT
    epsilon::Float64
end

function PowerFlowRecourse(
    polar::PolarFormRecourse{T, VI, VT, MT};
    epsilon=1e-4,
    alpha=nothing,
) where {T, VI, VT, MT}
    @assert polar.ncustoms > 0
    SMT = default_sparse_matrix(polar.device)
    k = nblocks(polar)

    pf = polar.network
    ngen = pf.ngen
    nbus = pf.nbus
    gen = pf.gen2bus
    npv = length(pf.pv)
    npq = length(pf.pq)
    @assert npv + npq + 1 == nbus

    # Assemble matrices
    Cg_tot = sparse(gen, 1:ngen, ones(ngen), nbus, ngen)
    Cd_tot = spdiagm(nbus, nbus, ones(nbus)) # Identity matrix
    Cg = -[Cg_tot[[pf.ref; pf.pv], :] ; spzeros(2*npq, ngen)]
    M_tot = PS.get_basis_matrix(polar.network)
    M = -M_tot[[pf.ref; pf.pv; pf.pq; nbus .+ pf.pq], :]
    # constant term
    Cdp = [Cd_tot[[pf.ref; pf.pv ; pf.pq], :]; spzeros(npq, nbus)]
    Cdq = [spzeros(nbus, nbus) ; Cd_tot[pf.pq, :]]

    M   = _blockdiag(M, k)
    Cg  = _blockdiag(Cg, k)
    Cdp = _blockdiag(Cdp, k)
    Cdq = _blockdiag(Cdq, k)

    # Response ratio (by default dispatch recourse evenly)
    if isnothing(alpha)
        alpha = ones(ngen) ./ ngen
    end
    # Bounds
    pgmin, pgmax = PS.bounds(polar.network, PS.Generators(), PS.ActivePower())

    return PowerFlowRecourse{VT, SMT}(M, Cg, Cdp, Cdq, pgmin, pgmax, alpha, epsilon)
end

Base.length(func::PowerFlowRecourse) = size(func.M, 1)

function _softmin(x1, x2, ϵ)
    xmax = max(x1, x2)
    xmin = min(x1, x2)
    return (xmin - ϵ * log1p(exp((xmin - xmax) / ϵ)))
end

# Smooth approximation of max(pmin, min(p, pmax))
# (numerically stable version)
function smooth_response(p, pmin, pmax, ϵ)
    @assert pmax - ϵ >= pmin
    threshold = 100.0
    if p >= pmax + threshold * ϵ
        return pmax
    elseif p >= 0.5 * (pmax + pmin)
        return _softmin(p, pmax, ϵ)
    elseif p >= (pmin - threshold * ϵ)
        return -_softmin(-p, -pmin, ϵ)
    else
        return pmin
    end
end

function (func::PowerFlowRecourse)(cons::AbstractArray, stack::AbstractNetworkStack)
    ngen = length(stack.pgen)
    fill!(cons, 0.0)
    # Constant terms
    mul!(cons, func.Cdp, stack.pload, 1.0, 1.0)
    mul!(cons, func.Cdq, stack.qload, 1.0, 1.0)
    # Variable terms
    mul!(cons, func.M, stack.ψ, 1.0, 1.0)

    Δ = stack.vuser[1]
    recourse_pgen = similar(cons, ngen)
    recourse_pgen .= smooth_response.(stack.pgen .+ Δ .* func.alpha, func.pgmin, func.pgmax, Ref(func.epsilon))
    mul!(cons, func.Cg, recourse_pgen, 1.0, 1.0)
    return
end

function bounds(polar::AbstractPolarFormulation{T,VI,VT,MT}, func::PowerFlowRecourse) where {T,VI,VT,MT}
    m = length(func)
    return (fill!(VT(undef, m), zero(T)) , fill!(VT(undef, m), zero(T)))
end

function Base.show(io::IO, func::PowerFlowRecourse)
    print(io, "PowerFlowRecourse (AbstractExpression)")
end
