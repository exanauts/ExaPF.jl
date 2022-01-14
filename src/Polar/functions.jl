

abstract type AbstractStack{VT} end


function Base.copyto!(stack::AbstractStack{VT}, map::AbstractVector{Int}, src::VT) where VT
    @assert length(map) == length(src)
    for i in eachindex(map)
        stack.input[map[i]] = src[i]
    end
end

function Base.copyto!(dest::VT, stack::AbstractStack{VT}, map::AbstractVector{Int}) where VT
    @assert length(map) == length(dest)
    for i in eachindex(map)
        dest[i] = stack.input[map[i]]
    end
end

struct NetworkStack{VT,NT} <: AbstractStack{VT}
    # INPUT
    input::VT
    vmag::VT # voltage magnitudes
    vang::VT # voltage angles
    pgen::VT # active power generations
    # INTERMEDIATE
    ψ::VT    # nonlinear basis ψ(vmag, vang)
    intermediate::NT
end

function NetworkStack(nbus, ngen, nlines, VT)
    input = VT(undef, 2*nbus + ngen) ; fill!(input, 0.0)
    # Wrap directly array x to avoid dealing with views
    p0 = pointer(input)
    vmag = unsafe_wrap(VT, p0, nbus)
    p1 = pointer(input, nbus+1)
    vang = unsafe_wrap(VT, p1, nbus)
    p2 = pointer(input, 2*nbus+1)
    pgen = unsafe_wrap(VT, p2, ngen)

    # Basis function
    ψ = VT(undef, 2*nlines + nbus) ; fill!(ψ, 0.0)
    # Intermediate expressions to avoid unecessary allocations
    intermediate = (
        c = VT(undef, ngen),     # buffer for costs
        sfp = VT(undef, nlines), # buffer for line-flow
        sfq = VT(undef, nlines), # buffer for line-flow
        stp = VT(undef, nlines), # buffer for line-flow
        stq = VT(undef, nlines), # buffer for line-flow
        ∂edge_vm_fr = VT(undef, nlines), # buffer for basis
        ∂edge_vm_to = VT(undef, nlines), # buffer for basis
        ∂edge_va_fr = VT(undef, nlines), # buffer for basis
        ∂edge_va_to = VT(undef, nlines), # buffer for basis
    )

    return NetworkStack(input, vmag, vang, pgen, ψ, intermediate)
end

function init!(polar::PolarForm, stack::NetworkStack)
    vmag = abs.(polar.network.vbus)
    vang = angle.(polar.network.vbus)
    pg = get(polar.network, PS.ActivePower())

    copyto!(stack.vmag, vmag)
    copyto!(stack.vang, vang)
    copyto!(stack.pgen, pg)
end

function NetworkStack(polar::PolarForm{T,VI,VT,MT}) where {T,VI,VT,MT}
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    nlines = get(polar, PS.NumberOfLines())
    stack = NetworkStack(nbus, ngen, nlines, VT)
    init!(polar, stack)
    return stack
end

function Base.empty!(state::NetworkStack)
    fill!(state.vmag, 0.0)
    fill!(state.vang, 0.0)
    fill!(state.pgen, 0.0)
    fill!(state.ψ, 0.0)
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


voltage(buf::NetworkStack) = buf.vmag .* exp.(im .* buf.vang)
voltage_host(buf::NetworkStack) = voltage(buf) |> Array


#=
    Generic expression
=#

abstract type AbstractExpression end

function (expr::AbstractExpression)(stack::AbstractStack)
    m = length(expr)
    output = similar(stack.input, m)
    expr(output, stack)
    return output
end

#=
    PolarBasis
=#

struct PolarBasis{VI, MT} <: AbstractExpression
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

    @inbounds begin
        if i <= nlines
            ℓ = i
            fr_bus = f[ℓ]
            to_bus = t[ℓ]
            Δθ = vang[fr_bus, j] - vang[to_bus, j]
            cosθ = cos(Δθ)
            cons[i, j] = vmag[fr_bus, j] * vmag[to_bus, j] * cosθ
        elseif i <= 2 * nlines
            ℓ = i - nlines
            fr_bus = f[ℓ]
            to_bus = t[ℓ]
            Δθ = vang[fr_bus, j] - vang[to_bus, j]
            sinθ = sin(Δθ)
            cons[i, j] = vmag[fr_bus, j] * vmag[to_bus, j] * sinθ
        elseif i <= 2 * nlines + nbus
            b = i - 2 * nlines
            cons[i, j] = vmag[b, j] * vmag[b, j]
        end
    end
end

function (func::PolarBasis)(output, stack::NetworkStack)
    ev = basis_kernel!(func.device)(
        output, stack.vmag, stack.vang,
        func.f, func.t, func.nlines, func.nbus,
        ndrange=(length(func), 1), dependencies=Event(func.device)
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

    @inbounds begin
        if i <= nlines
            ℓ = i
            fr_bus = f[ℓ]
            to_bus = t[ℓ]
            Δθ = vang[fr_bus, j] - vang[to_bus, j]
            cosθ = cos(Δθ)
            sinθ = sin(Δθ)

            adj_vang_fr[i] += -vmag[fr_bus, j] * vmag[to_bus, j] * sinθ * ∂cons[ℓ, j]
            adj_vang_fr[i] +=  vmag[fr_bus, j] * vmag[to_bus, j] * cosθ * ∂cons[ℓ+nlines, j]
            adj_vang_to[i] +=  vmag[fr_bus, j] * vmag[to_bus, j] * sinθ * ∂cons[ℓ, j]
            adj_vang_to[i] -=  vmag[fr_bus, j] * vmag[to_bus, j] * cosθ * ∂cons[ℓ+nlines, j]

            adj_vmag_fr[i] +=  vmag[to_bus, j] * cosθ * ∂cons[ℓ, j]
            adj_vmag_fr[i] += vmag[to_bus, j] * sinθ * ∂cons[ℓ+nlines, j]

            adj_vmag_to[i] +=  vmag[fr_bus, j] * cosθ * ∂cons[ℓ, j]
            adj_vmag_to[i] += vmag[fr_bus, j] * sinθ * ∂cons[ℓ+nlines, j]
        else i <= nlines + nbus
            b = i - nlines
            adj_vmag[b, j] += 2.0 * vmag[b, j] * ∂cons[b+2*nlines, j]
        end
    end
end

function adjoint!(func::PolarBasis, ∂state::NetworkStack, state::NetworkStack, ∂v)
    nl = func.nlines
    nb = func.nbus
    f = func.f
    t = func.t

    fill!(∂state.intermediate.∂edge_vm_fr , 0.0)
    fill!(∂state.intermediate.∂edge_vm_to , 0.0)
    fill!(∂state.intermediate.∂edge_va_fr , 0.0)
    fill!(∂state.intermediate.∂edge_va_to , 0.0)

    # Accumulate on edges
    ndrange = (nl+nb, 1)
    ev = adj_basis_kernel!(func.device)(
        ∂v,
        ∂state.vmag,
        ∂state.intermediate.∂edge_vm_fr,
        ∂state.intermediate.∂edge_vm_to,
        ∂state.intermediate.∂edge_va_fr,
        ∂state.intermediate.∂edge_va_to,
        state.vmag, state.vang, f, t, nl, nb,
        ndrange=ndrange, dependencies=Event(func.device),
    )
    wait(ev)

    # Accumulate on nodes
    Cf = func.Cf
    Ct = func.Ct
    mul!(∂state.vmag, Cf, ∂state.intermediate.∂edge_vm_fr, 1.0, 1.0)
    mul!(∂state.vmag, Ct, ∂state.intermediate.∂edge_vm_to, 1.0, 1.0)
    mul!(∂state.vang, Cf, ∂state.intermediate.∂edge_va_fr, 1.0, 1.0)
    mul!(∂state.vang, Ct, ∂state.intermediate.∂edge_va_to, 1.0, 1.0)
    return
end


#=
    CostFunction
=#

struct CostFunction{VT, MT} <: AbstractExpression
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
    return CostFunction{VT, SMT}(ref_gen, M, c0, c1, c2)
end

Base.length(::CostFunction) = 1

function (func::CostFunction)(output, state)
    costs = state.intermediate.c
    # Update pgen_ref
    state.pgen[func.gen_ref] .= 0.0
    mul!(state.pgen, func.M, state.ψ, 1.0, 1.0)
    costs .= func.c0 .+ func.c1 .* state.pgen .+ func.c2 .* state.pgen.^2
    CUDA.@allowscalar output[1] = sum(costs)
    return
end

function adjoint!(func::CostFunction, ∂state, state, ∂v)
    ∂state.pgen .+= ∂v .* (func.c1 .+ 2.0 .* func.c2 .* state.pgen)
    mul!(∂state.ψ, func.M', ∂state.pgen, 1.0, 1.0)
    return
end


@doc raw"""
    PowerFlowBalance

Subset of the power injection in the network
corresponding to ``(p_{inj}^{pv}, p_{inj}^{pq}, q_{inj}^{pq})``.
They are associated to the function

```math
g(x, u) = 0 .
```
introduced in the documentation.

In detail, the function encodes the active balance equations at
PV and PQ nodes, and the reactive balance equations at PQ nodes:
```math
\begin{aligned}
    p_i &= v_i \sum_{j}^{n} v_j (g_{ij}\cos{(\theta_i - \theta_j)} + b_{ij}\sin{(\theta_i - \theta_j})) \,, &
    ∀ i ∈ \{PV, PQ\} \\
    q_i &= v_i \sum_{j}^{n} v_j (g_{ij}\sin{(\theta_i - \theta_j)} - b_{ij}\cos{(\theta_i - \theta_j})) \,. &
    ∀ i ∈ \{PQ\}
\end{aligned}
```
"""
struct PowerFlowBalance{VT, MT} <: AbstractExpression
    M::MT
    Cg::MT
    τ::VT
end

function PowerFlowBalance(polar::PolarForm{T, VI, VT, MT}) where {T, VI, VT, MT}
    SMT = default_sparse_matrix(polar.device)

    pf = polar.network
    ngen = pf.ngen
    nbus = pf.nbus
    gen = pf.gen2bus
    pv = pf.pv
    npq = length(pf.pq)

    # Assemble matrices
    Cg_tot = sparse(gen, 1:ngen, ones(ngen), nbus, ngen)
    Cg = -[Cg_tot[pv, :] ; spzeros(2*npq, ngen)] |> SMT
    M_tot = PS.get_basis_matrix(polar.network)
    M = -M_tot[[pf.pv; pf.pq; nbus .+ pf.pq], :] |> SMT

    # constant term
    pload = PS.get(polar.network, PS.ActiveLoad())
    qload = PS.get(polar.network, PS.ReactiveLoad())
    τ = [pload[pf.pv]; pload[pf.pq]; qload[pf.pq]] |> VT

    return PowerFlowBalance{VT, SMT}(M, Cg, τ)
end

Base.length(func::PowerFlowBalance) = length(func.τ)

function bounds(polar::PolarForm{T,VI,VT,MT}, func::PowerFlowBalance) where {T,VI,VT,MT}
    m = length(func)
    return (fill!(VT(undef, m), zero(T)) , fill!(VT(undef, m), zero(T)))
end

function (func::PowerFlowBalance)(cons, state)
    cons .= func.τ
    mul!(cons, func.M, state.ψ, 1.0, 1.0)
    mul!(cons, func.Cg, state.pgen, 1.0, 1.0)
    return
end

function adjoint!(func::PowerFlowBalance, ∂state, state, ∂v)
    mul!(∂state.ψ, func.M', ∂v, 1.0, 1.0)
    mul!(∂state.pgen, func.Cg', ∂v, 1.0, 1.0)
    return
end


"""
    VoltageMagnitudePQ

Bounds the voltage magnitudes at PQ nodes:
```math
v_{pq}^♭ ≤ v_{pq} ≤ v_{pq}^♯ .
```

## Note
The constraints on the voltage magnitudes at PV nodes ``v_{pv}``
are taken into account when bounding the control ``u``.

"""
struct VoltageMagnitudePQ <: AbstractExpression
    pq::Vector{Int}

end
VoltageMagnitudePQ(polar::PolarForm) = VoltageMagnitudePQ(polar.network.pq)

Base.length(func::VoltageMagnitudePQ) = length(func.pq)

function bounds(polar::PolarForm{T,VI,VT,MT}, func::VoltageMagnitudePQ) where {T,VI,VT,MT}
    v_min, v_max = PS.bounds(polar.network, PS.Buses(), PS.VoltageMagnitude())
    return convert(VT, v_min[func.pq]), convert(VT, v_max[func.pq])
end

function (func::VoltageMagnitudePQ)(cons, state)
    cons .= state.vmag[func.pq]
end

function adjoint!(func::VoltageMagnitudePQ, ∂state, state, ∂v)
    ∂state.vmag[func.pq] .+= ∂v
end

"""
    PowerGenerationBounds

Constraints on the **active power production**
and on the **reactive power production** at the generators
that are not already taken into account in the bound constraints.
```math
p_g^♭ ≤ p_g ≤ p_g^♯  ;
q_g^♭ ≤ q_g ≤ q_g^♯  .
```
"""
struct PowerGenerationBounds{VT, MT} <: AbstractExpression
    M::MT
    τ::VT
end

function PowerGenerationBounds(polar::PolarForm{T, VI, VT, MT}) where {T, VI, VT, MT}
    SMT = default_sparse_matrix(polar.device)
    pf = polar.network
    nbus = pf.nbus
    M_tot = PS.get_basis_matrix(pf)

    M = -M_tot[[pf.ref; nbus .+ pf.ref; nbus .+ pf.pv], :]

    pload = PS.get(polar.network, PS.ActiveLoad())
    qload = PS.get(polar.network, PS.ReactiveLoad())
    τ = [pload[pf.ref]; qload[pf.ref]; qload[pf.pv]]

    return PowerGenerationBounds{VT, SMT}(M, τ)
end

Base.length(func::PowerGenerationBounds) = length(func.τ)

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
    return (
        convert(VT, [Cgp * p_min; Cgq * q_min]),
        convert(VT, [Cgp * p_max; Cgq * q_max]),
    )
end

function (func::PowerGenerationBounds)(cons, state)
    cons .= func.τ
    mul!(cons, func.M, state.ψ, 1.0, 1.0)
    return
end

function adjoint!(func::PowerGenerationBounds, ∂state, state, ∂v)
    mul!(∂state.ψ, func.M', ∂v, 1.0, 1.0)
    return
end


"""
    LineFlows

Thermal limit constraints porting on the lines of the network.

"""
struct LineFlows{VT, MT} <: AbstractExpression
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

function bounds(polar::PolarForm{T,VI,VT,MT}, func::LineFlows) where {T,VI,VT,MT}
    f_min, f_max = PS.bounds(polar.network, PS.Lines(), PS.ActivePower())
    return convert(VT, [f_min; f_min]), convert(VT, [f_max; f_max])
end

function (func::LineFlows)(cons::AbstractVector, state::NetworkStack{VT,S}) where {VT<:AbstractVector, S}
    sfp = state.intermediate.sfp::VT
    sfq = state.intermediate.sfq::VT
    stp = state.intermediate.stp::VT
    stq = state.intermediate.stq::VT

    mul!(sfp, func.Lfp, state.ψ)
    mul!(sfq, func.Lfq, state.ψ)
    mul!(stp, func.Ltp, state.ψ)
    mul!(stq, func.Ltq, state.ψ)
    cons[1:func.nlines] .= sfp.^2 .+ sfq.^2
    cons[1+func.nlines:2*func.nlines] .= stp.^2 .+ stq.^2
    return
end

function adjoint!(func::LineFlows, ∂state, state, ∂v)
    nlines = func.nlines
    sfp = ∂state.intermediate.sfp
    sfq = ∂state.intermediate.sfq
    stp = ∂state.intermediate.stp
    stq = ∂state.intermediate.stq
    mul!(sfp, func.Lfp, state.ψ)
    mul!(sfq, func.Lfq, state.ψ)
    mul!(stp, func.Ltp, state.ψ)
    mul!(stq, func.Ltq, state.ψ)

    @views begin
        sfp .*= ∂v[1:nlines]
        sfq .*= ∂v[1:nlines]
        stp .*= ∂v[1+nlines:2*nlines]
        stq .*= ∂v[1+nlines:2*nlines]
    end

    # Accumulate adjoint
    mul!(∂state.ψ, func.Lfp', sfp, 2.0, 1.0)
    mul!(∂state.ψ, func.Lfq', sfq, 2.0, 1.0)
    mul!(∂state.ψ, func.Ltp', stp, 2.0, 1.0)
    mul!(∂state.ψ, func.Ltq', stq, 2.0, 1.0)

    return
end

# Concatenate expressions together
struct MultiExpressions <: AbstractExpression
    exprs::Vector{AbstractExpression}
end

Base.length(func::MultiExpressions) = sum(length.(func.exprs))

function (func::MultiExpressions)(output, state)
    k = 0
    for expr in func.exprs
        m = length(expr)
        y = view(output, k+1:k+m)
        expr(y, state)
        k += m
    end
end

function adjoint!(func::MultiExpressions, ∂state, state, ∂v)
    k = 0
    for expr in func.exprs
        m = length(expr)
        y = view(∂v, k+1:k+m)
        adjoint!(expr, ∂state, state, y)
        k += m
    end
end

function bounds(polar::PolarForm{T, VI, VT, MT}, func::MultiExpressions) where {T, VI, VT, MT}
    m = length(func)
    g_min = zeros(m)
    g_max = zeros(m)
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

struct ComposedExpressions{Expr1<:PolarBasis, Expr2} <: AbstractExpression
    inner::Expr1
    outer::Expr2
end

function (func::ComposedExpressions)(output, state)
    func.inner(state.ψ, state)  # Evaluate basis
    func.outer(output, state)   # Evaluate expression
end

function adjoint!(func::ComposedExpressions, ∂state, state, ∂v)
    adjoint!(func.outer, ∂state, state, ∂v)
    adjoint!(func.inner, ∂state, state, ∂state.ψ)
end

# Overload ∘ operator
Base.ComposedFunction(g::AbstractExpression, f::PolarBasis) = ComposedExpressions(f, g)
Base.length(func::ComposedExpressions) = length(func.outer)
bounds(polar, func::ComposedExpressions) = bounds(polar, func.outer)

