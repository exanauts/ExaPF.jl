
struct PolarFormRecourse{T, IT, VT, MT} <: AbstractPolarFormulation{T, IT, VT, MT}
    network::PS.PowerNetwork
    device::KA.Device
    k::Int
    ncustoms::Int  # custom variables defined by user
end

function PolarFormRecourse(pf::PS.PowerNetwork, device::KA.CPU, k::Int)
    ngen = PS.get(pf, PS.NumberOfGenerators())
    ncustoms = (ngen + 1)
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
    @assert length(alpha) == ngen
    # Bounds
    _pgmin, _pgmax = PS.bounds(polar.network, PS.Generators(), PS.ActivePower())
    pgmin = repeat(_pgmin, k)
    pgmax = repeat(_pgmax, k)

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
    k = nblocks(stack)
    ngen = length(stack.pgen)
    fill!(cons, 0.0)
    # Constant terms
    mul!(cons, func.Cdp, stack.pload, 1.0, 1.0)
    mul!(cons, func.Cdq, stack.qload, 1.0, 1.0)
    # Variable terms
    mul!(cons, func.M, stack.ψ, 1.0, 1.0)

    Δ = view(stack.vuser, 1:k)
    pgen_setpoint = view(stack.vuser, k+1:k+ngen)
    p1 = reshape(pgen_setpoint, div(ngen, k), k)
    p2 = reshape(stack.pgen, div(ngen, k), k)
    # raw recourse
    p2 .= p1 .+ Δ' .* func.alpha
    # smoothened recourse
    stack.pgen .= smooth_response.(stack.pgen, func.pgmin, func.pgmax, Ref(func.epsilon))
    mul!(cons, func.Cg, stack.pgen, 1.0, 1.0)
    return
end

function bounds(polar::AbstractPolarFormulation{T,VI,VT,MT}, func::PowerFlowRecourse) where {T,VI,VT,MT}
    m = length(func)
    return (fill!(VT(undef, m), zero(T)) , fill!(VT(undef, m), zero(T)))
end

function Base.show(io::IO, func::PowerFlowRecourse)
    print(io, "PowerFlowRecourse (AbstractExpression)")
end


"""
    ReactivePowerBounds
"""
struct ReactivePowerBounds{VT, MT} <: AutoDiff.AbstractExpression
    M::MT
    Cdq::MT
end

function ReactivePowerBounds(polar::PolarFormRecourse{T, VI, VT, MT}) where {T, VI, VT, MT}
    SMT = default_sparse_matrix(polar.device)
    pf = polar.network
    nbus = pf.nbus
    M_tot = PS.get_basis_matrix(pf)
    ns = length(pf.ref) + length(pf.pv)

    M = -M_tot[[nbus .+ pf.ref; nbus .+ pf.pv], :]
    Cd_tot = spdiagm(nbus, nbus, ones(nbus)) # Identity matrix
    Cdq = Cd_tot[[pf.ref ; pf.pv], :]

    M = _blockdiag(M, nblocks(polar))
    Cdq = _blockdiag(Cdq, nblocks(polar))

    return ReactivePowerBounds{VT, SMT}(M, Cdq)
end

Base.length(func::ReactivePowerBounds) = size(func.M, 1)

function (func::ReactivePowerBounds)(cons::AbstractArray, stack::AbstractNetworkStack)
    fill!(cons, 0.0)
    # Constant terms
    mul!(cons, func.Cdq, stack.qload, 1.0, 1.0)
    # Variable terms
    mul!(cons, func.M, stack.ψ, 1.0, 1.0)
    return
end

function adjoint!(func::ReactivePowerBounds, ∂stack, stack, ∂v)
    mul!(∂stack.ψ, func.M', ∂v, 1.0, 1.0)
    return
end

function bounds(polar::PolarFormRecourse{T,VI,VT,MT}, func::ReactivePowerBounds) where {T,VI,VT,MT}
    pf = polar.network
    ngen = pf.ngen
    nbus = pf.nbus
    ref, pv = pf.ref, pf.pv
    # Build incidence matrix
    Cg = sparse(pf.gen2bus, 1:ngen, ones(ngen), nbus, ngen)
    Cgq = Cg[[ref ; pv], :]
    # Get original bounds
    q_min, q_max = PS.bounds(polar.network, PS.Generators(), PS.ReactivePower())
    # Aggregate bounds on ref and pv nodes
    lb = Cgq * q_min
    ub = Cgq * q_max
    return (
        convert(VT, repeat(lb, nblocks(polar))),
        convert(VT, repeat(ub, nblocks(polar))),
    )
end

function Base.show(io::IO, func::ReactivePowerBounds)
    print(io, "ReactivePowerBounds (AbstractExpression)")
end


#=
    Mapping
=#

function mapping(polar::PolarFormRecourse, ::Control, k::Int=1)
    pf = polar.network
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    ngen = polar.network.ngen
    ref, pv = pf.ref, pf.pv
    genidx = 1:ngen
    nu = (length(pv) + length(ref) + length(genidx)) * k
    mapu = zeros(Int, nu)

    shift_mag = 0
    shift_pgen = (2 * nbus + 1) * k
    index = 1
    for i in 1:k
        for j in ref
            mapu[index] = j + (i-1) * nbus + shift_mag
            index += 1
        end
        for j in pv
            mapu[index] = j + (i-1) * nbus + shift_mag
            index += 1
        end
        for j in genidx
            mapu[index] = j + (i-1) * ngen + shift_pgen
            index += 1
        end
    end
    return mapu
end

function mapping(polar::PolarFormRecourse, ::State, k::Int=1)
    pf = polar.network
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    ndelta = 1
    ref, pv, pq = pf.ref, pf.pv, pf.pq
    nx = (length(pv) + 2*length(pq) + ndelta) * k
    mapx = zeros(Int, nx)

    shift_ang = k * nbus
    shift_mag = 0
    shift_delta = (2 * nbus + ngen) * k
    index = 1
    for i in 1:k
        for j in [pv; pq]
            mapx[index] = j + (i-1) * nbus + shift_ang
            index += 1
        end
        for j in pq
            mapx[index] = j + (i-1) * nbus + shift_mag
            index += 1
        end
        # delta
        mapx[index] = 1 + (i-1) * ndelta + shift_delta
        index += 1
    end
    return mapx
end

function mapping(polar::PolarFormRecourse, ::AllVariables, k::Int=1)
    pf = polar.network
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    ndelta = 1
    ref, pv, pq = pf.ref, pf.pv, pf.pq
    genidx = collect(1:ngen)
    nx = (length(pv) + 2*length(pq)) * k
    nu = (length(pv) + length(ref) + length(genidx)) * k
    mapxu = zeros(Int, nx+nu)

    shift_mag = 0
    shift_ang = k * nbus
    shift_pgen = (2 * nbus + ngen + ndelta) * k
    index = 1
    for i in 1:k
        # / x
        for j in [pv; pq]
            mapxu[index] = j + (i-1) * nbus + shift_ang
            index += 1
        end
        for j in pq
            mapxu[index] = j + (i-1) * nbus + shift_mag
            index += 1
        end
        # delta
        mapx[index] = j + (i-1) * ndelta + shift_delta
        index += 1
        # / u
        for j in [ref; pv]
            mapxu[index] = j + (i-1) * nbus + shift_mag
            index += 1
        end
        for j in genidx
            mapxu[index] = j + (i-1) * ngen + shift_pgen
            index += 1
        end
    end
    return mapxu
end

#=
    Stack
=#

function init!(polar::PolarFormRecourse, stack::NetworkStack; update_loads=true)
    vmag = get(polar.network, PS.VoltageMagnitude())
    vang = get(polar.network, PS.VoltageAngle())
    pgen = get(polar.network, PS.ActivePower())
    pload = get(polar.network, PS.ActiveLoad())
    qload = get(polar.network, PS.ReactiveLoad())

    nbus = length(vmag)
    ngen = length(pgen)
    nload = length(pload)

    for s in 1:nblocks(stack)
        i = (s - 1) * nbus  + 1
        copyto!(stack.vmag, i, vmag, 1, nbus)
        copyto!(stack.vang, i, vang, 1, nbus)

        i = (s - 1) * ngen  + 1
        copyto!(stack.pgen, i, pgen, 1, ngen)

        i = (s - 1) * ngen  + nblocks(stack) + 1
        copyto!(stack.vuser, i, pgen, 1, ngen)

        if update_loads
            i = (s - 1) * nload  + 1
            copyto!(stack.pload, i, pload, 1, nload)
            copyto!(stack.qload, i, qload, 1, nload)
        end
    end
end

function bounds(polar::PolarFormRecourse{T, VI, VT, MT}, stack::NetworkStack) where {T, VI, VT, MT}
    nbus = polar.network.nbus
    vmag_min, vmag_max = PS.bounds(polar.network, PS.Buses(), PS.VoltageMagnitude())
    vang_min, vang_max = fill(-Inf, nbus), fill(Inf, nbus)
    pgen_min, pgen_max = PS.bounds(polar.network, PS.Generators(), PS.ActivePower())
    delta_min, delta_max = -Inf, Inf

    lb = [
        repeat(vmag_min, nblocks(polar));
        repeat(vang_min, nblocks(polar));
        repeat(pgen_min, nblocks(polar));
        fill(delta_min, nblocks(polar));
        repeat(pgen_min, nblocks(polar));
    ]
    ub = [
        repeat(vmag_max, nblocks(polar));
        repeat(vang_max, nblocks(polar));
        repeat(pgen_max, nblocks(polar));
        fill(delta_max, nblocks(polar));
        repeat(pgen_max, nblocks(polar));
    ]
    return convert(VT, lb), convert(VT, ub)
end

