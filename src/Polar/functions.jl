
function Base.copyto!(stack::AutoDiff.AbstractStack{VT}, map::AbstractVector{Int}, src::AbstractVector) where {VT}
    @assert length(map) == length(src)
    for i in eachindex(map)
        stack.input[map[i]] = src[i]
    end
end

function Base.copyto!(dest::AbstractVector, stack::AutoDiff.AbstractStack{VT}, map::AbstractVector{Int}) where {VT}
    @assert length(map) == length(dest)
    for i in eachindex(map)
        dest[i] = stack.input[map[i]]
    end
end

abstract type AbstractNetworkStack{VT} <: AutoDiff.AbstractStack{VT} end

"""
    NetworkStack <: AbstractStack
    NetworkStack(polar::PolarForm)
    NetworkStack(nbus::Int, ngen::Int, nlines::Int, VT::Type)

Store the variables associated to the polar formulation.
The variables are stored in the field `input`, ordered as follows
```
    input = [vmag ; vang ; pgen]
```
The object stores also intermediate variables needed
in the expression tree, such as the LKMR basis `ψ`.

### Notes

The NetworkStack can be instantiated on the host or on
the target device.


"""
struct NetworkStack{VT,VD,NT} <: AbstractNetworkStack{VT}
    # INPUT
    input::VD
    vmag::VD # voltage magnitudes
    vang::VD # voltage angles
    pgen::VD # active power generations
    # INTERMEDIATE
    ψ::VD    # nonlinear basis ψ(vmag, vang)
    intermediate::NT
    # Parameters
    params::VT
    pload::VT
    qload::VT
end

function NetworkStack(nbus, ngen, nlines, VT, VD)
    input = VD(undef, 2*nbus + ngen) ; fill!(input, 0.0)
    # Wrap directly array x to avoid dealing with views
    p0 = pointer(input)
    vmag = unsafe_wrap(VD, p0, nbus)
    p1 = pointer(input, nbus+1)
    vang = unsafe_wrap(VD, p1, nbus)
    p2 = pointer(input, 2*nbus+1)
    pgen = unsafe_wrap(VD, p2, ngen)

    # Basis function
    ψ = VD(undef, 2*nlines + nbus) ; fill!(ψ, 0.0)
    # Intermediate expressions to avoid unecessary allocations
    intermediate = (
        c = VD(undef, ngen),     # buffer for costs
        sfp = VD(undef, nlines), # buffer for line-flow
        sfq = VD(undef, nlines), # buffer for line-flow
        stp = VD(undef, nlines), # buffer for line-flow
        stq = VD(undef, nlines), # buffer for line-flow
        ∂edge_vm_fr = VD(undef, nlines), # buffer for basis
        ∂edge_vm_to = VD(undef, nlines), # buffer for basis
        ∂edge_va_fr = VD(undef, nlines), # buffer for basis
        ∂edge_va_to = VD(undef, nlines), # buffer for basis
    )

    # Parameters: loads
    params = VT(undef, 2*nbus) ; fill!(params, 0.0)
    p0 = pointer(params)
    pload = unsafe_wrap(VT, p0, nbus)
    p1 = pointer(params, nbus+1)
    qload = unsafe_wrap(VT, p1, nbus)

    return NetworkStack(input, vmag, vang, pgen, ψ, intermediate, params, pload, qload)
end

function Base.show(io::IO, stack::NetworkStack)
    print(io, "$(length(stack.input))-elements NetworkStack{$(typeof(stack.input))}")
end

nbatches(stack::NetworkStack) = 1

"""
    init!(polar::PolarForm, stack::NetworkStack)

Set `stack.input` with the initial values specified
in the base [`PS.PowerNetwork`](@ref) object.

"""
function init!(polar::PolarForm, stack::NetworkStack)
    copyto!(stack.vmag, get(polar.network, PS.VoltageMagnitude()))
    copyto!(stack.vang, get(polar.network, PS.VoltageAngle()))
    copyto!(stack.pgen, get(polar.network, PS.ActivePower()))
    copyto!(stack.pload, get(polar.network, PS.ActiveLoad()))
    copyto!(stack.qload, get(polar.network, PS.ReactiveLoad()))
end

function NetworkStack(polar::PolarForm{T,VI,VT,MT}) where {T,VI,VT,MT}
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    nlines = get(polar, PS.NumberOfLines())
    stack = NetworkStack(nbus, ngen, nlines, VT, VT)
    init!(polar, stack)
    return stack
end

function Base.empty!(stack::NetworkStack)
    fill!(stack.vmag, 0.0)
    fill!(stack.vang, 0.0)
    fill!(stack.pgen, 0.0)
    fill!(stack.ψ, 0.0)
    fill!(stack.pload, 0.0)
    fill!(stack.qload, 0.0)
    return
end

function bounds(polar::PolarForm{T, VI, VT, MT}, stack::NetworkStack) where {T, VI, VT, MT}
    nbus = polar.network.nbus
    vmag_min, vmag_max = PS.bounds(polar.network, PS.Buses(), PS.VoltageMagnitude())
    vang_min, vang_max = fill(-Inf, nbus), fill(Inf, nbus)
    pgen_min, pgen_max = PS.bounds(polar.network, PS.Generators(), PS.ActivePower())

    lb = [vmag_min; vang_min; pgen_min]
    ub = [vmag_max; vang_max; pgen_max]
    return convert(VT, lb), convert(VT, ub)
end

"Get complex voltage from `NetworkStack`."
voltage(buf::NetworkStack) = buf.vmag .* exp.(im .* buf.vang)
voltage_host(buf::NetworkStack) = voltage(buf) |> Array


#=
    PolarBasis
=#

@doc raw"""
    PolarBasis{VI, MT} <: AbstractExpression
    PolarBasis(polar::PolarForm)

Implement the LKMR nonlinear basis. Takes as
input the voltage magnitudes `vmag` and the voltage
angles `vang` and returns
```math
    \begin{aligned}
        & \psi_\ell^C(v, \theta) = v^f  v^t  \cos(\theta_f - \theta_t) \quad \forall \ell = 1, \cdots, n_\ell \\
        & \psi_\ell^S(v, \theta) = v^f  v^t  \sin(\theta_f - \theta_t) \quad \forall \ell = 1, \cdots, n_\ell \\
        & \psi_k(v, \theta) = v_k^2 \quad \forall k = 1, \cdots, n_b
    \end{aligned}
```

**Dimension:** `2 * n_lines + n_bus`

### Complexity
`3 n_lines + n_bus` mul, `n_lines` `cos` and `n_lines` `sin`

"""
struct PolarBasis{VI, MT} <: AutoDiff.AbstractExpression
    nbus::Int
    nlines::Int
    f::VI
    t::VI
    Cf::MT
    Ct::MT
    device::KA.Device
end

function PolarBasis(polar::PolarForm{T, VI, VT, MT}) where {T, VI, VT, MT}
    SMT = default_sparse_matrix(polar.device)
    nlines = PS.get(polar.network, PS.NumberOfLines())
    # Assemble matrix
    pf = polar.network
    nbus = pf.nbus
    lines = pf.lines
    f = lines.from_buses
    t = lines.to_buses

    Cf = sparse(f, 1:nlines, ones(nlines), nbus, nlines)
    Ct = sparse(t, 1:nlines, ones(nlines), nbus, nlines)
    Cf = Cf |> SMT
    Ct = Ct |> SMT

    return PolarBasis{VI, SMT}(nbus, nlines, f, t, Cf, Ct, polar.device)
end

Base.length(func::PolarBasis) = func.nbus + 2 * func.nlines

# update basis
@kernel function basis_kernel!(
    cons, @Const(vmag), @Const(vang), @Const(f), @Const(t), nlines, nbus,
)
    i, j = @index(Global, NTuple)
    shift_cons = (j-1) * (nbus + 2*nlines)
    shift_bus  = (j-1) * nbus

    @inbounds begin
        if i <= nlines
            ℓ = i
            fr_bus = f[ℓ]
            to_bus = t[ℓ]
            Δθ = vang[fr_bus + shift_bus] - vang[to_bus + shift_bus]
            cosθ = cos(Δθ)
            cons[i + shift_cons] = vmag[fr_bus + shift_bus] * vmag[to_bus + shift_bus] * cosθ
        elseif i <= 2 * nlines
            ℓ = i - nlines
            fr_bus = f[ℓ]
            to_bus = t[ℓ]
            Δθ = vang[fr_bus + shift_bus] - vang[to_bus + shift_bus]
            sinθ = sin(Δθ)
            cons[i + shift_cons] = vmag[fr_bus + shift_bus] * vmag[to_bus + shift_bus] * sinθ
        elseif i <= 2 * nlines + nbus
            b = i - 2 * nlines
            cons[i + shift_cons] = vmag[b + shift_bus] * vmag[b + shift_bus]
        end
    end
end

function (func::PolarBasis)(output, stack::AbstractNetworkStack)
    ndrange = (length(func), nbatches(stack))
    ev = basis_kernel!(func.device)(
        output, stack.vmag, stack.vang,
        func.f, func.t, func.nlines, func.nbus,
        ndrange=ndrange, dependencies=Event(func.device)
    )
    wait(ev)
    return
end

@kernel function adj_basis_kernel!(
    ∂cons, adj_vmag, adj_vmag_fr, adj_vmag_to,
    adj_vang_fr, adj_vang_to,
    @Const(vmag), @Const(vang), @Const(f), @Const(t), nlines, nbus,
)
    i, j = @index(Global, NTuple)
    shift_cons = (j-1) * (nbus + 2*nlines)
    shift_bus  = (j-1) * nbus

    @inbounds begin
        if i <= nlines
            ℓ = i
            fr_bus = f[ℓ]
            to_bus = t[ℓ]
            Δθ = vang[fr_bus + shift_bus] - vang[to_bus + shift_bus]
            cosθ = cos(Δθ)
            sinθ = sin(Δθ)

            adj_vang_fr[i + shift_bus] += -vmag[fr_bus + shift_bus] * vmag[to_bus + shift_bus] * sinθ * ∂cons[ℓ + shift_cons]
            adj_vang_fr[i + shift_bus] +=  vmag[fr_bus + shift_bus] * vmag[to_bus + shift_bus] * cosθ * ∂cons[ℓ+nlines + shift_cons]
            adj_vang_to[i + shift_bus] +=  vmag[fr_bus + shift_bus] * vmag[to_bus + shift_bus] * sinθ * ∂cons[ℓ + shift_cons]
            adj_vang_to[i + shift_bus] -=  vmag[fr_bus + shift_bus] * vmag[to_bus + shift_bus] * cosθ * ∂cons[ℓ+nlines + shift_cons]

            adj_vmag_fr[i + shift_bus] +=  vmag[to_bus + shift_bus] * cosθ * ∂cons[ℓ + shift_cons]
            adj_vmag_fr[i + shift_bus] += vmag[to_bus + shift_bus] * sinθ * ∂cons[ℓ+nlines + shift_cons]

            adj_vmag_to[i + shift_bus] +=  vmag[fr_bus + shift_bus] * cosθ * ∂cons[ℓ + shift_cons]
            adj_vmag_to[i + shift_bus] += vmag[fr_bus + shift_bus] * sinθ * ∂cons[ℓ+nlines + shift_cons]
        else i <= nlines + nbus
            b = i - nlines
            adj_vmag[b + shift_bus] += 2.0 * vmag[b + shift_bus] * ∂cons[b+2*nlines + shift_cons]
        end
    end
end

function adjoint!(func::PolarBasis, ∂stack::AbstractNetworkStack, stack::AbstractNetworkStack, ∂v)
    nl = func.nlines
    nb = func.nbus
    f = func.f
    t = func.t

    fill!(∂stack.intermediate.∂edge_vm_fr , 0.0)
    fill!(∂stack.intermediate.∂edge_vm_to , 0.0)
    fill!(∂stack.intermediate.∂edge_va_fr , 0.0)
    fill!(∂stack.intermediate.∂edge_va_to , 0.0)

    # Accumulate on edges
    ndrange = (nl+nb, nbatches(stack))
    ev = adj_basis_kernel!(func.device)(
        ∂v,
        ∂stack.vmag,
        ∂stack.intermediate.∂edge_vm_fr,
        ∂stack.intermediate.∂edge_vm_to,
        ∂stack.intermediate.∂edge_va_fr,
        ∂stack.intermediate.∂edge_va_to,
        stack.vmag, stack.vang, f, t, nl, nb,
        ndrange=ndrange, dependencies=Event(func.device),
    )
    wait(ev)

    # Accumulate on nodes
    Cf = func.Cf
    Ct = func.Ct
    blockmul!(∂stack.vmag, Cf, ∂stack.intermediate.∂edge_vm_fr, 1.0, 1.0)
    blockmul!(∂stack.vmag, Ct, ∂stack.intermediate.∂edge_vm_to, 1.0, 1.0)
    blockmul!(∂stack.vang, Cf, ∂stack.intermediate.∂edge_va_fr, 1.0, 1.0)
    blockmul!(∂stack.vang, Ct, ∂stack.intermediate.∂edge_va_to, 1.0, 1.0)
    return
end

function Base.show(io::IO, func::PolarBasis)
    print(io, "PolarBasis (AbstractExpression)")
end


#=
    CostFunction
=#

@doc raw"""
    CostFunction{VT, MT} <: AutoDiff.AbstractExpression
    CostFunction(polar)

Implement the quadratic cost function for OPF
```math
    ∑_{g=1}^{n_g} c_{2,g} p_g^2 + c_{1,g} p_g + c_{0,g}
```

**Dimension:** `1`

### Complexity
`1` SpMV, `1` `sum`

"""
struct CostFunction{VT, MT} <: AutoDiff.AbstractExpression
    ref::Vector{Int}
    gen_ref::Vector{Int}
    M::MT
    c0::VT
    c1::VT
    c2::VT
end

function CostFunction(polar::PolarForm{T, VI, VT, MT}) where {T, VI, VT, MT}
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    SMT = default_sparse_matrix(polar.device)
    # Load indexing
    ref = polar.network.ref
    gen2bus = polar.network.gen2bus
    if length(ref) > 1
        error("Too many generators are affected to the slack nodes")
    end
    ref_gen = Int[findfirst(isequal(ref[1]), gen2bus)]

    # Gen-bus incidence matrix
    Cg = sparse(ref_gen, ref, ones(1), ngen, 2 * nbus)
    # Assemble matrix
    M_tot = PS.get_basis_matrix(polar.network)
    M = - Cg * M_tot |> SMT

    # coefficients
    coefs = PS.get_costs_coefficients(polar.network)
    c0 = @view coefs[:, 2]
    c1 = @view coefs[:, 3]
    c2 = @view coefs[:, 4]

    return CostFunction{VT, SMT}(ref, ref_gen, M, c0, c1, c2)
end

Base.length(::CostFunction) = 1

function (func::CostFunction)(output::AbstractArray, stack::NetworkStack)
    costs = stack.intermediate.c
    # Update pgen_ref
    stack.pgen[func.gen_ref] .= stack.pload[func.ref]
    mul!(stack.pgen, func.M, stack.ψ, 1.0, 1.0)
    costs .= func.c0 .+ func.c1 .* stack.pgen .+ func.c2 .* stack.pgen.^2
    CUDA.@allowscalar output[1] = sum(costs)
    return
end

function adjoint!(func::CostFunction, ∂stack, stack, ∂v)
    ∂stack.pgen .+= ∂v .* (func.c1 .+ 2.0 .* func.c2 .* stack.pgen)
    mul!(∂stack.ψ, func.M', ∂stack.pgen, 1.0, 1.0)
    return
end

function Base.show(io::IO, func::CostFunction)
    print(io, "CostFunction (AbstractExpression)")
end


#=
    PowerFlowBalance
=#

@doc raw"""
    PowerFlowBalance{VT, MT}
    PowerFlowBalance(polar)

Implement a subset of the power injection
corresponding to ``(p_{inj}^{pv}, p_{inj}^{pq}, q_{inj}^{pq})``.
The function encodes the active balance equations at
PV and PQ nodes, and the reactive balance equations at PQ nodes:
```math
\begin{aligned}
    p_i &= v_i \sum_{j}^{n} v_j (g_{ij}\cos{(\theta_i - \theta_j)} + b_{ij}\sin{(\theta_i - \theta_j})) \,, &
    ∀ i ∈ \{PV, PQ\} \\
    q_i &= v_i \sum_{j}^{n} v_j (g_{ij}\sin{(\theta_i - \theta_j)} - b_{ij}\cos{(\theta_i - \theta_j})) \,. &
    ∀ i ∈ \{PQ\}
\end{aligned}
```

**Dimension:** `n_{pv} + 2 * n_{pq}`

### Complexity
`2` SpMV

"""
struct PowerFlowBalance{VT, MT} <: AutoDiff.AbstractExpression
    M::MT
    Cg::MT
    Cdp::MT
    Cdq::MT
end

function PowerFlowBalance(polar::PolarForm{T, VI, VT, MT}) where {T, VI, VT, MT}
    SMT = default_sparse_matrix(polar.device)

    pf = polar.network
    ngen = pf.ngen
    nbus = pf.nbus
    gen = pf.gen2bus
    npv = length(pf.pv)
    npq = length(pf.pq)

    # Assemble matrices
    Cg_tot = sparse(gen, 1:ngen, ones(ngen), nbus, ngen)
    Cd_tot = spdiagm(nbus, nbus, ones(nbus)) # Identity matrix
    Cg = -[Cg_tot[pf.pv, :] ; spzeros(2*npq, ngen)] |> SMT
    M_tot = PS.get_basis_matrix(polar.network)
    M = -M_tot[[pf.pv; pf.pq; nbus .+ pf.pq], :] |> SMT

    # constant term
    Cdp = [Cd_tot[[pf.pv ; pf.pq], :]; spzeros(npq, nbus)] |> SMT
    Cdq = [spzeros(npq+npv, nbus) ; Cd_tot[pf.pq, :]] |> SMT
    return PowerFlowBalance{VT, SMT}(M, Cg, Cdp, Cdq)
end

Base.length(func::PowerFlowBalance) = size(func.M, 1)

function (func::PowerFlowBalance)(cons::AbstractArray, stack::AbstractNetworkStack)
    fill!(cons, 0.0)
    # Constant terms
    blockmul!(cons, func.Cdp, stack.pload, 1.0, 1.0)
    blockmul!(cons, func.Cdq, stack.qload, 1.0, 1.0)
    # Variable terms
    blockmul!(cons, func.M, stack.ψ, 1.0, 1.0)
    blockmul!(cons, func.Cg, stack.pgen, 1.0, 1.0)
    return
end

function adjoint!(func::PowerFlowBalance, ∂stack, stack, ∂v)
    blockmul!(∂stack.ψ, func.M', ∂v, 1.0, 1.0)
    blockmul!(∂stack.pgen, func.Cg', ∂v, 1.0, 1.0)
    return
end

function bounds(polar::PolarForm{T,VI,VT,MT}, func::PowerFlowBalance) where {T,VI,VT,MT}
    m = length(func)
    return (fill!(VT(undef, m), zero(T)) , fill!(VT(undef, m), zero(T)))
end

function Base.show(io::IO, func::PowerFlowBalance)
    print(io, "PowerFlowBalance (AbstractExpression)")
end


"""
    VoltageMagnitudeBounds

Implement the bounds on voltage magnitudes not
taken into account in the bound constraints. In
the reduced space, this is associated to the
the voltage magnitudes at PQ nodes:
```math
v_{pq}^♭ ≤ v_{pq} ≤ v_{pq}^♯ .
```

**Dimension:** `n_pq`

### Complexity
`1` copyto

### Note
In the reduced space, the constraints on the voltage magnitudes at PV nodes ``v_{pv}``
are taken into account when bounding the control ``u``.

"""
struct VoltageMagnitudeBounds <: AutoDiff.AbstractExpression
    pq::Vector{Int}
end
VoltageMagnitudeBounds(polar::PolarForm) = VoltageMagnitudeBounds(polar.network.pq)

Base.length(func::VoltageMagnitudeBounds) = length(func.pq)

function (func::VoltageMagnitudeBounds)(cons::AbstractArray, stack::AbstractNetworkStack)
    cons .= stack.vmag[func.pq]
end

function adjoint!(func::VoltageMagnitudeBounds, ∂stack, stack, ∂v)
    ∂stack.vmag[func.pq] .+= ∂v
end

function bounds(polar::PolarForm{T,VI,VT,MT}, func::VoltageMagnitudeBounds) where {T,VI,VT,MT}
    v_min, v_max = PS.bounds(polar.network, PS.Buses(), PS.VoltageMagnitude())
    return convert(VT, v_min[func.pq]), convert(VT, v_max[func.pq])
end

function Base.show(io::IO, func::VoltageMagnitudeBounds)
    print(io, "VoltageMagnitudeBounds (AbstractExpression)")
end


"""
    PowerGenerationBounds{VT, MT}
    PowerGenerationBounds(polar)

Constraints on the active power productions
and on the reactive power productions
that are not already taken into account in the bound constraints.
In the reduced space, that amounts to
```math
p_{g,ref}^♭ ≤ p_{g,ref} ≤ p_{g,ref}^♯  ;
C_g q_g^♭ ≤ C_g q_g ≤ C_g q_g^♯  .
```

**Dimension:** `n_pv + 2 n_ref`

### Complexity
`1` copyto, `1` SpMV

"""
struct PowerGenerationBounds{VT, MT} <: AutoDiff.AbstractExpression
    M::MT
    Cdp::MT
    Cdq::MT
end

function PowerGenerationBounds(polar::PolarForm{T, VI, VT, MT}) where {T, VI, VT, MT}
    SMT = default_sparse_matrix(polar.device)
    pf = polar.network
    nbus = pf.nbus
    M_tot = PS.get_basis_matrix(pf)
    ns = length(pf.ref) + length(pf.pv)

    M = -M_tot[[pf.ref; nbus .+ pf.ref; nbus .+ pf.pv], :]
    Cd_tot = spdiagm(nbus, nbus, ones(nbus)) # Identity matrix

    Cdp = [Cd_tot[pf.ref, :] ; spzeros(ns, nbus)]
    Cdq = [spzeros(length(pf.ref), nbus) ; Cd_tot[[pf.ref ; pf.pv], :]]
    return PowerGenerationBounds{VT, SMT}(M, Cdp, Cdq)
end

Base.length(func::PowerGenerationBounds) = size(func.M, 1)

function (func::PowerGenerationBounds)(cons::AbstractArray, stack::AbstractNetworkStack)
    fill!(cons, 0.0)
    # Constant terms
    blockmul!(cons, func.Cdp, stack.pload, 1.0, 1.0)
    blockmul!(cons, func.Cdq, stack.qload, 1.0, 1.0)
    # Variable terms
    blockmul!(cons, func.M, stack.ψ, 1.0, 1.0)
    return
end

function adjoint!(func::PowerGenerationBounds, ∂stack, stack, ∂v)
    blockmul!(∂stack.ψ, func.M', ∂v, 1.0, 1.0)
    return
end

function bounds(polar::PolarForm{T,VI,VT,MT}, func::PowerGenerationBounds) where {T,VI,VT,MT}
    pf = polar.network
    ngen = pf.ngen
    nbus = pf.nbus
    ref, pv = pf.ref, pf.pv
    # Build incidence matrix
    Cg = sparse(pf.gen2bus, 1:ngen, ones(ngen), nbus, ngen)
    Cgp = Cg[ref, :]
    Cgq = Cg[[ref ; pv], :]
    # Get original bounds
    p_min, p_max = PS.bounds(polar.network, PS.Generators(), PS.ActivePower())
    q_min, q_max = PS.bounds(polar.network, PS.Generators(), PS.ReactivePower())
    # Aggregate bounds on ref and pv nodes
    lb = [Cgp * p_min; Cgq * q_min]
    ub = [Cgp * p_max; Cgq * q_max]
    return (
        convert(VT, lb),
        convert(VT, ub),
    )
end

function Base.show(io::IO, func::PowerGenerationBounds)
    print(io, "PowerGenerationBounds (AbstractExpression)")
end


"""
    LineFlows{VT, MT}
    LineFlows(polar)

Implement thermal limit constraints on the lines of the network.

**Dimension:** `2 * n_lines`

### Complexity
`4` SpMV, `4 * n_lines` quadratic, `2 * n_lines` add

"""
struct LineFlows{VT, MT} <: AutoDiff.AbstractExpression
    nlines::Int
    Lfp::MT
    Lfq::MT
    Ltp::MT
    Ltq::MT
end

function LineFlows(polar::PolarForm{T,VI,VT,MT}) where {T,VI,VT,MT}
    SMT = default_sparse_matrix(polar.device)
    nlines = get(polar, PS.NumberOfLines())
    Lfp, Lfq, Ltp, Ltq = PS.get_line_flow_matrices(polar.network)
    return LineFlows{VT,SMT}(nlines, Lfp, Lfq, Ltp, Ltq)
end

Base.length(func::LineFlows) = 2 * func.nlines

function (func::LineFlows)(cons::AbstractVector, stack::NetworkStack{VT,VD,S}) where {VT<:AbstractVector, VD<:AbstractVector, S}
    sfp = stack.intermediate.sfp::VD
    sfq = stack.intermediate.sfq::VD
    stp = stack.intermediate.stp::VD
    stq = stack.intermediate.stq::VD

    mul!(sfp, func.Lfp, stack.ψ)
    mul!(sfq, func.Lfq, stack.ψ)
    mul!(stp, func.Ltp, stack.ψ)
    mul!(stq, func.Ltq, stack.ψ)
    cons[1:func.nlines] .= sfp.^2 .+ sfq.^2
    cons[1+func.nlines:2*func.nlines] .= stp.^2 .+ stq.^2
    return
end

function adjoint!(func::LineFlows, ∂stack, stack, ∂v)
    nlines = func.nlines
    sfp = ∂stack.intermediate.sfp
    sfq = ∂stack.intermediate.sfq
    stp = ∂stack.intermediate.stp
    stq = ∂stack.intermediate.stq
    mul!(sfp, func.Lfp, stack.ψ)
    mul!(sfq, func.Lfq, stack.ψ)
    mul!(stp, func.Ltp, stack.ψ)
    mul!(stq, func.Ltq, stack.ψ)

    @views begin
        sfp .*= ∂v[1:nlines]
        sfq .*= ∂v[1:nlines]
        stp .*= ∂v[1+nlines:2*nlines]
        stq .*= ∂v[1+nlines:2*nlines]
    end

    # Accumulate adjoint
    mul!(∂stack.ψ, func.Lfp', sfp, 2.0, 1.0)
    mul!(∂stack.ψ, func.Lfq', sfq, 2.0, 1.0)
    mul!(∂stack.ψ, func.Ltp', stp, 2.0, 1.0)
    mul!(∂stack.ψ, func.Ltq', stq, 2.0, 1.0)

    return
end

function bounds(polar::PolarForm{T,VI,VT,MT}, func::LineFlows) where {T,VI,VT,MT}
    f_min, f_max = PS.bounds(polar.network, PS.Lines(), PS.ActivePower())
    return convert(VT, [f_min; f_min]), convert(VT, [f_max; f_max])
end

function Base.show(io::IO, func::LineFlows)
    print(io, "LineFlows (AbstractExpression)")
end


#=
    MultiExpressions
=#

"""
    MultiExpressions <: AbstractExpression

Implement expressions concatenation. Takes as
input a vector of expressions `[expr1,...,exprN]`
and concatenate them in a single expression `mexpr`, such
that
```
    mexpr(x) = [expr1(x) ; expr2(x) ; ... ; exprN(x)]

```

"""
struct MultiExpressions <: AutoDiff.AbstractExpression
    exprs::Vector{AutoDiff.AbstractExpression}
end

Base.length(func::MultiExpressions) = sum(length.(func.exprs))

function (func::MultiExpressions)(output::AbstractArray, stack::AutoDiff.AbstractStack)
    k = 0
    for expr in func.exprs
        m = length(expr)
        y = view(output, k+1:k+m)
        expr(y, stack)
        k += m
    end
end

function adjoint!(func::MultiExpressions, ∂stack, stack, ∂v)
    k = 0
    for expr in func.exprs
        m = length(expr)
        y = view(∂v, k+1:k+m)
        adjoint!(expr, ∂stack, stack, y)
        k += m
    end
end

function bounds(polar::PolarForm{T, VI, VT, MT}, func::MultiExpressions) where {T, VI, VT, MT}
    m = length(func)
    g_min = VT(undef, m)
    g_max = VT(undef, m)
    k = 0
    for expr in func.exprs
        m = length(expr)
        l, u = bounds(polar, expr)
        g_min[k+1:k+m] .= l
        g_max[k+1:k+m] .= u
        k += m
    end
    return (
        convert(VT, g_min),
        convert(VT, g_max),
    )
end

#=
    ComposedExpressions
=#

"""
    ComposedExpressions{Expr1<:PolarBasis, Expr2} <: AbstractExpression

Implement expression composition. Takes as input two expressions
`expr1` and `expr2` and returns a composed expression `cexpr` such
that
```
    cexpr(x) = expr2 ∘ expr1(x)

### Notes
Currently, only [`PolarBasis`](@ref) is supported for `expr1`.
"""
struct ComposedExpressions{Expr1<:PolarBasis, Expr2} <: AutoDiff.AbstractExpression
    inner::Expr1
    outer::Expr2
end
Base.length(func::ComposedExpressions) = length(func.outer)

function (func::ComposedExpressions)(output::AbstractArray, stack::AutoDiff.AbstractStack)
    func.inner(stack.ψ, stack)  # Evaluate basis
    func.outer(output, stack)   # Evaluate expression
end

function adjoint!(func::ComposedExpressions, ∂stack, stack, ∂v)
    adjoint!(func.outer, ∂stack, stack, ∂v)
    adjoint!(func.inner, ∂stack, stack, ∂stack.ψ)
end

bounds(polar, func::ComposedExpressions) = bounds(polar, func.outer)
# Overload ∘ operator
Base.ComposedFunction(g::AutoDiff.AbstractExpression, f::PolarBasis) = ComposedExpressions(f, g)

