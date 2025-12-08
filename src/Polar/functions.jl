
#=
    PolarBasis
=#

@doc raw"""
    PolarBasis{VI, MT} <: AbstractExpression
    PolarBasis(polar::AbstractPolarFormulation)

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

### Examples
```jldoctest; setup=:(using ExaPF)
julia> polar = ExaPF.load_polar("case9");

julia> stack = ExaPF.NetworkStack(polar);

julia> basis = ExaPF.PolarBasis(polar)
PolarBasis (AbstractExpression)

julia> basis(stack)
27-element Vector{Float64}:
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 0.0
 ⋮
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0

```
"""
struct PolarBasis{VI, MT} <: AutoDiff.AbstractExpression
    nbus::Int
    nlines::Int
    f::VI
    t::VI
    Cf::MT
    Ct::MT
    backend::KA.Backend
end

function PolarBasis(polar::AbstractPolarFormulation{T, VI, VT, MT}) where {T, VI, VT, MT}
    SMT = default_sparse_matrix(polar.backend)
    k = nblocks(polar)
    nlines = PS.get(polar.network, PS.NumberOfLines())
    # Assemble matrix
    pf = polar.network
    nbus = pf.nbus
    lines = pf.lines
    f = lines.from_buses
    t = lines.to_buses

    Cf = sparse(f, 1:nlines, ones(nlines), nbus, nlines)
    Ct = sparse(t, 1:nlines, ones(nlines), nbus, nlines)
    Cf = _blockdiag(Cf, k) |> SMT
    Ct = _blockdiag(Ct, k) |> SMT

    return PolarBasis{VI, SMT}(nbus, nlines, f, t, Cf, Ct, polar.backend)
end

Base.length(func::PolarBasis) = (func.nbus + 2 * func.nlines) * div(size(func.Cf, 1), func.nbus)

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
    ndrange = (func.nbus + 2 * func.nlines, nblocks(stack))
    basis_kernel!(func.backend)(
        output, stack.vmag, stack.vang,
        func.f, func.t, func.nlines, func.nbus,
        ndrange=ndrange
    )
    KA.synchronize(func.backend)
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
    shift_lines  = (j-1) * nlines

    @inbounds begin
        if i <= nlines
            ℓ = i
            fr_bus = f[ℓ]
            to_bus = t[ℓ]
            Δθ = vang[fr_bus + shift_bus] - vang[to_bus + shift_bus]
            cosθ = cos(Δθ)
            sinθ = sin(Δθ)

            adj_vang_fr[i + shift_lines] -= vmag[fr_bus + shift_bus] * vmag[to_bus + shift_bus] * sinθ * ∂cons[ℓ + shift_cons]
            adj_vang_fr[i + shift_lines] += vmag[fr_bus + shift_bus] * vmag[to_bus + shift_bus] * cosθ * ∂cons[ℓ+nlines + shift_cons]
            adj_vang_to[i + shift_lines] += vmag[fr_bus + shift_bus] * vmag[to_bus + shift_bus] * sinθ * ∂cons[ℓ + shift_cons]
            adj_vang_to[i + shift_lines] -= vmag[fr_bus + shift_bus] * vmag[to_bus + shift_bus] * cosθ * ∂cons[ℓ+nlines + shift_cons]

            adj_vmag_fr[i + shift_lines] += vmag[to_bus + shift_bus] * cosθ * ∂cons[ℓ + shift_cons]
            adj_vmag_fr[i + shift_lines] += vmag[to_bus + shift_bus] * sinθ * ∂cons[ℓ+nlines + shift_cons]

            adj_vmag_to[i + shift_lines] += vmag[fr_bus + shift_bus] * cosθ * ∂cons[ℓ + shift_cons]
            adj_vmag_to[i + shift_lines] += vmag[fr_bus + shift_bus] * sinθ * ∂cons[ℓ+nlines + shift_cons]
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
    ndrange = (nl+nb, nblocks(stack))
    adj_basis_kernel!(func.backend)(
        ∂v,
        ∂stack.vmag,
        ∂stack.intermediate.∂edge_vm_fr,
        ∂stack.intermediate.∂edge_vm_to,
        ∂stack.intermediate.∂edge_va_fr,
        ∂stack.intermediate.∂edge_va_to,
        stack.vmag, stack.vang, f, t, nl, nb,
        ndrange=ndrange,
    )
    KA.synchronize(func.backend)

    # Accumulate on nodes
    Cf = func.Cf
    Ct = func.Ct
    mul!(∂stack.vmag, Cf, ∂stack.intermediate.∂edge_vm_fr, 1.0, 1.0)
    mul!(∂stack.vmag, Ct, ∂stack.intermediate.∂edge_vm_to, 1.0, 1.0)
    mul!(∂stack.vang, Cf, ∂stack.intermediate.∂edge_va_fr, 1.0, 1.0)
    mul!(∂stack.vang, Ct, ∂stack.intermediate.∂edge_va_to, 1.0, 1.0)
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
Require composition with [`PolarBasis`](@ref) to evaluate
the cost of the reference generator.

**Dimension:** `1`

### Complexity
`1` SpMV, `1` `sum`

### Examples
```jldoctest; setup=:(using ExaPF)
julia> polar = ExaPF.load_polar("case9");

julia> stack = ExaPF.NetworkStack(polar);

julia> cost = ExaPF.CostFunction(polar) ∘ ExaPF.PolarBasis(polar);

julia> cost(stack)
1-element Vector{Float64}:
 4509.0275

```
"""
struct CostFunction{VT, MT} <: AutoDiff.AbstractExpression
    ref::Vector{Int}
    gen_ref::Vector{Int}
    M::MT
    N::MT
    c0::VT
    c1::VT
    c2::VT
    backend::KA.Backend
end

function CostFunction(polar::AbstractPolarFormulation{T, VI, VT, MT}) where {T, VI, VT, MT}
    SMT = default_sparse_matrix(polar.backend)
    k = nblocks(polar)
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
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
    M = - Cg * M_tot
    M = _blockdiag(M, k)

    N = sparse(ref_gen, ref, ones(1), ngen, nbus)
    N = _blockdiag(N, k)

    # coefficients
    coefs = PS.get_costs_coefficients(polar.network)
    c0 = @view coefs[:, 2]
    c1 = @view coefs[:, 3]
    c2 = @view coefs[:, 4]

    return CostFunction{VT, SMT}(ref, ref_gen, M, N, c0, c1, c2, polar.backend)
end

Base.length(func::CostFunction) = div(size(func.N, 1), length(func.c0))

@kernel function _quadratic_cost_kernel(cost, pgen, c0, c1, c2, ngen)
    i, j = @index(Global, NTuple)
    shift_gen  = (j-1) * ngen
    pg = pgen[i + shift_gen]
    cost[i + shift_gen] = c0[i] + c1[i] * pg + c2[i] * pg^2
end

function (func::CostFunction)(output::AbstractArray, stack::AbstractNetworkStack)
    ngen = length(func.c0)
    costs = stack.intermediate.c
    # Update pgen_ref
    pgen_m = reshape(stack.pgen, ngen, nblocks(stack))
    pgen_m[func.gen_ref, :] .= 0.0
    # Add load to pgen_ref
    mul!(stack.pgen, func.N, stack.pload, 1.0, 1.0)
    # Recompute power injection at ref node
    mul!(stack.pgen, func.M, stack.ψ, 1.0, 1.0)
    # Compute quadratic costs
    ndrange = (ngen, nblocks(stack))
    _quadratic_cost_kernel(func.backend)(
        costs, stack.pgen, func.c0, func.c1, func.c2, ngen;
        ndrange=ndrange,
    )
    KA.synchronize(func.backend)
    # Sum costs across all generators
    # sum!(output, reshape(costs, ngen, nblocks(stack))')
    output .= sum(reshape(costs, ngen, nblocks(stack))', dims=2)
    return
end

@kernel function _adj_quadratic_cost_kernel(adj_pgen, pgen, adj_v, c0, c1, c2, ngen)
    i, j = @index(Global, NTuple)
    shift_gen  = (j-1) * ngen
    pg = pgen[i + shift_gen]
    adj_pgen[i + shift_gen] += (c1[i] + 2.0 * c2[i] * pg) * adj_v[1]
end

function adjoint!(func::CostFunction, ∂stack, stack, ∂v)
    ngen = length(func.c0)
    ndrange = (ngen, nblocks(stack))
    _adj_quadratic_cost_kernel(func.backend)(
        ∂stack.pgen, stack.pgen, ∂v, func.c0, func.c1, func.c2, ngen;
        ndrange=ndrange,
    )
    KA.synchronize(func.backend)
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

Require composition with [`PolarBasis`](@ref).

**Dimension:** `n_pv + 2 * n_pq`

### Complexity
`2` SpMV

### Examples
```jldoctest; setup=:(using ExaPF)
julia> polar = ExaPF.load_polar("case9");

julia> stack = ExaPF.NetworkStack(polar);

julia> powerflow = ExaPF.PowerFlowBalance(polar) ∘ ExaPF.PolarBasis(polar);

julia> round.(powerflow(stack); digits=6)
14-element Vector{Float64}:
 -1.63
 -0.85
  0.0
  0.9
  0.0
  1.0
  0.0
  1.25
 -0.167
  0.042
 -0.2835
  0.171
 -0.2275
  0.259

julia> run_pf(polar, stack); # solve powerflow equations

julia> isapprox(powerflow(stack), zeros(14); atol=1e-8)
true

```

"""
struct PowerFlowBalance{VT, MT} <: AutoDiff.AbstractExpression
    M::MT
    Cg::MT
    Cdp::MT
    Cdq::MT
end

function PowerFlowBalance(polar::AbstractPolarFormulation{T, VI, VT, MT}) where {T, VI, VT, MT}
    SMT = default_sparse_matrix(polar.backend)
    k = nblocks(polar)

    pf = polar.network
    ngen = pf.ngen
    nbus = pf.nbus
    gen = pf.gen2bus
    npv = length(pf.pv)
    npq = length(pf.pq)

    # Assemble matrices
    Cg_tot = sparse(gen, 1:ngen, ones(ngen), nbus, ngen)
    Cd_tot = spdiagm(nbus, nbus, ones(nbus)) # Identity matrix
    Cg = -[Cg_tot[pf.pv, :] ; spzeros(2*npq, ngen)]
    M_tot = PS.get_basis_matrix(polar.network)
    M = -M_tot[[pf.pv; pf.pq; nbus .+ pf.pq], :]
    # constant term
    Cdp = [Cd_tot[[pf.pv ; pf.pq], :]; spzeros(npq, nbus)]
    Cdq = [spzeros(npq+npv, nbus) ; Cd_tot[pf.pq, :]]

    M   = _blockdiag(M, k)
    Cg  = _blockdiag(Cg, k)
    Cdp = _blockdiag(Cdp, k)
    Cdq = _blockdiag(Cdq, k)

    return PowerFlowBalance{VT, SMT}(M, Cg, Cdp, Cdq)
end

Base.length(func::PowerFlowBalance) = size(func.M, 1)

function (func::PowerFlowBalance)(cons::AbstractArray, stack::AbstractNetworkStack)
    fill!(cons, 0.0)
    # Constant terms
    mul!(cons, func.Cdp, stack.pload, 1.0, 1.0)
    mul!(cons, func.Cdq, stack.qload, 1.0, 1.0)
    # Variable terms
    mul!(cons, func.M, stack.ψ, 1.0, 1.0)
    mul!(cons, func.Cg, stack.pgen, 1.0, 1.0)
    return
end

function adjoint!(func::PowerFlowBalance, ∂stack, stack, ∂v)
    mul!(∂stack.ψ, func.M', ∂v, 1.0, 1.0)
    mul!(∂stack.pgen, func.Cg', ∂v, 1.0, 1.0)
    return
end

function bounds(polar::AbstractPolarFormulation{T,VI,VT,MT}, func::PowerFlowBalance) where {T,VI,VT,MT}
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

### Examples
```jldoctest; setup=:(using ExaPF)
julia> polar = ExaPF.load_polar("case9");

julia> stack = ExaPF.NetworkStack(polar);

julia> voltage_pq = ExaPF.VoltageMagnitudeBounds(polar);

julia> voltage_pq(stack)
6-element Vector{Float64}:
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0

```

"""
struct VoltageMagnitudeBounds{SMT} <: AutoDiff.AbstractExpression
    Cpq::SMT
end
function VoltageMagnitudeBounds(polar::AbstractPolarFormulation)
    SMT = default_sparse_matrix(polar.backend)
    nbus = polar.network.nbus
    pq = polar.network.pq
    C = spdiagm(ones(nbus))
    Cpq = C[pq, :]
    Cpq = _blockdiag(Cpq, nblocks(polar)) |> SMT
    return VoltageMagnitudeBounds(Cpq)
end

Base.length(func::VoltageMagnitudeBounds) = size(func.Cpq, 1)

function (func::VoltageMagnitudeBounds)(cons::AbstractArray, stack::AbstractNetworkStack)
    mul!(cons, func.Cpq, stack.vmag, 1.0, 0.0)
end

function adjoint!(func::VoltageMagnitudeBounds, ∂stack, stack, ∂v)
    mul!(∂stack.vmag, func.Cpq', ∂v, 1.0, 1.0)
end

function bounds(polar::AbstractPolarFormulation{T,VI,VT,MT}, func::VoltageMagnitudeBounds) where {T,VI,VT,MT}
    v_min, v_max = PS.bounds(polar.network, PS.Buses(), PS.VoltageMagnitude())
    v_min = v_min |> VT
    v_max = v_max |> VT

    lb = func.Cpq * repeat(v_min, nblocks(polar))
    ub =  func.Cpq * repeat(v_max, nblocks(polar))
    return lb, ub
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
Require composition with [`PolarBasis`](@ref).

**Dimension:** `n_pv + 2 n_ref`

### Complexity
`1` copyto, `1` SpMV

### Examples
```jldoctest; setup=:(using ExaPF)
julia> polar = ExaPF.load_polar("case9");

julia> stack = ExaPF.NetworkStack(polar);

julia> run_pf(polar, stack); # solve powerflow equations

julia> power_generators = ExaPF.PowerGenerationBounds(polar) ∘ ExaPF.PolarBasis(polar);

julia> round.(power_generators(stack); digits=6)
4-element Vector{Float64}:
  0.719547
  0.24069
  0.144601
 -0.03649

```

"""
struct PowerGenerationBounds{VT, MT} <: AutoDiff.AbstractExpression
    M::MT
    Cdp::MT
    Cdq::MT
end

function PowerGenerationBounds(polar::AbstractPolarFormulation{T, VI, VT, MT}) where {T, VI, VT, MT}
    SMT = default_sparse_matrix(polar.backend)
    pf = polar.network
    nbus = pf.nbus
    M_tot = PS.get_basis_matrix(pf)
    ns = length(pf.ref) + length(pf.pv)

    M = -M_tot[[pf.ref; nbus .+ pf.ref; nbus .+ pf.pv], :]
    Cd_tot = spdiagm(nbus, nbus, ones(nbus)) # Identity matrix

    Cdp = [Cd_tot[pf.ref, :] ; spzeros(ns, nbus)]
    Cdq = [spzeros(length(pf.ref), nbus) ; Cd_tot[[pf.ref ; pf.pv], :]]

    M = _blockdiag(M, nblocks(polar))
    Cdp = _blockdiag(Cdp, nblocks(polar))
    Cdq = _blockdiag(Cdq, nblocks(polar))

    return PowerGenerationBounds{VT, SMT}(M, Cdp, Cdq)
end

Base.length(func::PowerGenerationBounds) = size(func.M, 1)

function (func::PowerGenerationBounds)(cons::AbstractArray, stack::AbstractNetworkStack)
    fill!(cons, 0.0)
    # Constant terms
    mul!(cons, func.Cdp, stack.pload, 1.0, 1.0)
    mul!(cons, func.Cdq, stack.qload, 1.0, 1.0)
    # Variable terms
    mul!(cons, func.M, stack.ψ, 1.0, 1.0)
    return
end

function adjoint!(func::PowerGenerationBounds, ∂stack, stack, ∂v)
    mul!(∂stack.ψ, func.M', ∂v, 1.0, 1.0)
    return
end

function bounds(polar::AbstractPolarFormulation{T,VI,VT,MT}, func::PowerGenerationBounds) where {T,VI,VT,MT}
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
        convert(VT, repeat(lb, nblocks(polar))),
        convert(VT, repeat(ub, nblocks(polar))),
    )
end

function Base.show(io::IO, func::PowerGenerationBounds)
    print(io, "PowerGenerationBounds (AbstractExpression)")
end


"""
    LineFlows{VT, MT}
    LineFlows(polar)

Implement thermal limit constraints on the lines of the network.

Require composition with [`PolarBasis`](@ref).

**Dimension:** `2 * n_lines`

### Complexity
`4` SpMV, `4 * n_lines` quadratic, `2 * n_lines` add

### Examples
```jldoctest; setup=:(using ExaPF)
julia> polar = ExaPF.load_polar("case9");

julia> stack = ExaPF.NetworkStack(polar);

julia> run_pf(polar, stack); # solve powerflow equations

julia> line_flows = ExaPF.LineFlows(polar) ∘ ExaPF.PolarBasis(polar);

julia> round.(line_flows(stack); digits=6)
18-element Vector{Float64}:
 0.575679
 0.094457
 0.379983
 0.723832
 0.060169
 0.588673
 2.657418
 0.748943
 0.295351
 0.560817
 0.112095
 0.38625
 0.728726
 0.117191
 0.585164
 2.67781
 0.726668
 0.215497

```

"""
struct LineFlows{VT, MT} <: AutoDiff.AbstractExpression
    nlines::Int
    Lfp::MT
    Lfq::MT
    Ltp::MT
    Ltq::MT
    backend::KA.Backend
end

function LineFlows(polar::AbstractPolarFormulation{T,VI,VT,MT}) where {T,VI,VT,MT}
    SMT = default_sparse_matrix(polar.backend)
    nlines = get(polar, PS.NumberOfLines())
    Lfp, Lfq, Ltp, Ltq = PS.get_line_flow_matrices(polar.network)
    return LineFlows{VT,SMT}(
        nlines,
        _blockdiag(Lfp, nblocks(polar)),
        _blockdiag(Lfq, nblocks(polar)),
        _blockdiag(Ltp, nblocks(polar)),
        _blockdiag(Ltq, nblocks(polar)),
        polar.backend,
    )
end

Base.length(func::LineFlows) = 2 * size(func.Lfp, 1)

@kernel function _line_flow_kernel(output, sfp, sfq, stp, stq, nlines)
    i, j = @index(Global, NTuple)
    shift_lines = (j-1) * nlines
    output[i + 2 * shift_lines] = sfp[i + shift_lines]^2 + sfq[i + shift_lines]^2
    output[i + nlines + 2 * shift_lines] = stp[i + shift_lines]^2 + stq[i + shift_lines]^2
end

function (func::LineFlows)(cons::AbstractVector, stack::AbstractNetworkStack)
    sfp = stack.intermediate.sfp
    sfq = stack.intermediate.sfq
    stp = stack.intermediate.stp
    stq = stack.intermediate.stq

    mul!(sfp, func.Lfp, stack.ψ, 1.0, 0.0)
    mul!(sfq, func.Lfq, stack.ψ, 1.0, 0.0)
    mul!(stp, func.Ltp, stack.ψ, 1.0, 0.0)
    mul!(stq, func.Ltq, stack.ψ, 1.0, 0.0)
    ndrange = (func.nlines, nblocks(stack))
    ev = _line_flow_kernel(func.backend)(
        cons, sfp, sfq, stp, stq, func.nlines;
        ndrange=ndrange,
    )
    KA.synchronize(func.backend)
    return
end

@kernel function _adj_line_flow_kernel(
    adj_sfp, adj_sfq, adj_stp, adj_stq,
    sfp, sfq, stp, stq, adj_v, nlines,
)
    i, j = @index(Global, NTuple)
    shift_lines = (j-1) * nlines
    adj_sfp[i + shift_lines] = 2.0 * sfp[i + shift_lines] * adj_v[i + 2 * shift_lines]
    adj_sfq[i + shift_lines] = 2.0 * sfq[i + shift_lines] * adj_v[i + 2 * shift_lines]
    adj_stp[i + shift_lines] = 2.0 * stp[i + shift_lines] * adj_v[i + nlines + 2 * shift_lines]
    adj_stq[i + shift_lines] = 2.0 * stq[i + shift_lines] * adj_v[i + nlines + 2 * shift_lines]
end

function adjoint!(func::LineFlows, ∂stack, stack, ∂v)
    nlines = func.nlines
    sfp = ∂stack.intermediate.sfp
    sfq = ∂stack.intermediate.sfq
    stp = ∂stack.intermediate.stp
    stq = ∂stack.intermediate.stq

    ndrange = (func.nlines, nblocks(stack))
    _adj_line_flow_kernel(func.backend)(
        sfp, sfq, stp, stq,
        stack.intermediate.sfp, stack.intermediate.sfq,
        stack.intermediate.stp, stack.intermediate.stq,
        ∂v, nlines;
        ndrange=ndrange,
    )
    KA.synchronize(func.backend)

    # Accumulate adjoint
    mul!(∂stack.ψ, func.Lfp', sfp, 1.0, 1.0)
    mul!(∂stack.ψ, func.Lfq', sfq, 1.0, 1.0)
    mul!(∂stack.ψ, func.Ltp', stp, 1.0, 1.0)
    mul!(∂stack.ψ, func.Ltq', stq, 1.0, 1.0)

    return
end

function bounds(polar::AbstractPolarFormulation{T,VI,VT,MT}, func::LineFlows) where {T,VI,VT,MT}
    f_min, f_max = PS.bounds(polar.network, PS.Lines(), PS.ActivePower())
    lb = [f_min; f_min]
    ub = [f_max; f_max]
    return (
        convert(VT, repeat(lb, nblocks(polar))),
        convert(VT, repeat(ub, nblocks(polar))),
    )
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
AutoDiff.has_multiple_expressions(func::MultiExpressions) = true
AutoDiff.get_slices(func::MultiExpressions) = length.(func.exprs)

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

function bounds(polar::AbstractPolarFormulation{T, VI, VT, MT}, func::MultiExpressions) where {T, VI, VT, MT}
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
AutoDiff.has_multiple_expressions(func::ComposedExpressions) = AutoDiff.has_multiple_expressions(func.outer)
AutoDiff.get_slices(func::ComposedExpressions) = AutoDiff.get_slices(func.outer)

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

