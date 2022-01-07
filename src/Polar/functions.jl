

abstract type AbstractStack end

struct NetworkStack{VT,NT} <: AbstractStack
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
    )

    return NetworkStack(input, vmag, vang, pgen, ψ, intermediate)
end

function NetworkStack(polar::PolarForm{T,VI,VT,MT}) where {T,VI,VT,MT}
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    nlines = get(polar, PS.NumberOfLines())

    stack = NetworkStack(nbus, ngen, nlines, VT)

    copyto!(stack.vmag, abs.(polar.network.vbus))
    copyto!(stack.vang, angle.(polar.network.vbus))
    copyto!(stack.pgen, get(polar.network, PS.ActivePower()))
    return stack
end

function Base.empty!(state::NetworkStack)
    fill!(state.vmag, 0.0)
    fill!(state.vang, 0.0)
    fill!(state.pgen, 0.0)
    fill!(state.ψ, 0.0)
    return
end


# update basis
function forward_eval_intermediate(polar::PolarForm, state::NetworkStack)
    _network_basis(polar, state.ψ, state.vmag, state.vang)
end

function reverse_eval_intermediate(polar::PolarForm, ∂state::NetworkStack, state::NetworkStack, intermediate)
    nl = PS.get(polar.network, PS.NumberOfLines())
    nb = PS.get(polar.network, PS.NumberOfBuses())
    top = polar.topology
    f = top.f_buses
    t = top.t_buses

    fill!(intermediate.∂edge_vm_fr , 0.0)
    fill!(intermediate.∂edge_vm_to , 0.0)
    fill!(intermediate.∂edge_va_fr , 0.0)
    fill!(intermediate.∂edge_va_to , 0.0)

    # Accumulate on edges
    ndrange = (nl+nb, size(∂state.vmag, 2))
    ev = adj_basis_kernel!(polar.device)(
        ∂state.ψ,
        ∂state.vmag,
        intermediate.∂edge_vm_fr,
        intermediate.∂edge_vm_to,
        intermediate.∂edge_va_fr,
        intermediate.∂edge_va_to,
        state.vmag, state.vang, f, t, nl, nb,
        ndrange=ndrange, dependencies=Event(polar.device),
    )
    wait(ev)

    # Accumulate on nodes
    Cf = intermediate.Cf
    Ct = intermediate.Ct
    mul!(∂state.vmag, Cf, intermediate.∂edge_vm_fr, 1.0, 1.0)
    mul!(∂state.vmag, Ct, intermediate.∂edge_vm_to, 1.0, 1.0)
    mul!(∂state.vang, Cf, intermediate.∂edge_va_fr, 1.0, 1.0)
    mul!(∂state.vang, Ct, intermediate.∂edge_va_to, 1.0, 1.0)
    return
end

#=
    Generic expression
=#

abstract type AbstractExpression end


include("first_order.jl")
include("second_order.jl")


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
    ngen = get(polar, PS.NumberOfGenerators())
    SMT = default_sparse_matrix(polar.device)
    # Load indexing
    ref = polar.network.ref
    ref_gen = polar.indexing.index_ref_to_gen
    # Assemble matrix
    M_tot = PS.get_basis_matrix(polar.network)
    M = -M_tot[ref, :] |> SMT

    # coefficients
    coefs = polar.costs_coefficients
    c0 = @view coefs[:, 2]
    c1 = @view coefs[:, 3]
    c2 = @view coefs[:, 4]
    return CostFunction{VT, SMT}(ref_gen, M, c0, c1, c2)
end

Base.length(::CostFunction) = 1

function (func::CostFunction)(state)
    costs = state.intermediate.c
    state.pgen[func.gen_ref] .= func.M * state.ψ
    costs .= func.c0 .+ func.c1 .* state.pgen .+ func.c2 .* state.pgen.^2
    return sum(costs)
end

function (func::CostFunction)(output, state)
    output[1] = func(state)
    return
end

function adjoint!(func::CostFunction, ∂state, state, ∂v)
    ∂state.pgen .+= ∂v .* (func.c1 .+ 2.0 .* func.c2 .* state.pgen)
    ∂state.ψ .+= func.M' * ∂state.pgen[func.gen_ref]
    return
end


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


struct VoltageMagnitudePQ <: AbstractExpression
    pq::Vector{Int}

end
VoltageMagnitudePQ(polar::PolarForm) = VoltageMagnitudePQ(polar.network.pq)

Base.length(func::VoltageMagnitudePQ) = length(func.pq)

function (func::VoltageMagnitudePQ)(cons, state)
    cons .= state.vmag[func.pq]
end

function adjoint!(func::VoltageMagnitudePQ, ∂state, state, ∂v)
    ∂state.vmag[func.pq] .+= ∂v
end


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

function (func::PowerGenerationBounds)(cons, state)
    cons .= func.τ .+ func.M * state.ψ
    return
end

function adjoint!(func::PowerGenerationBounds, ∂state, state, ∂v)
    mul!(∂state.ψ, func.M', ∂v, 1.0, 1.0)
    return
end


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

    sfp .*= ∂v[1:nlines]
    sfq .*= ∂v[1:nlines]
    stp .*= ∂v[1+nlines:2*nlines]
    stq .*= ∂v[1+nlines:2*nlines]

    # Accumulate adjoint
    mul!(∂state.ψ, func.Lfp', sfp, 2.0, 1.0)
    mul!(∂state.ψ, func.Lfq', sfq, 2.0, 1.0)
    mul!(∂state.ψ, func.Ltp', stp, 2.0, 1.0)
    mul!(∂state.ψ, func.Ltq', stq, 2.0, 1.0)

    return
end

# Aggregate expressions together
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

include("legacy.jl")

