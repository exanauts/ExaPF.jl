
struct HessianProd{Model, Func, VT, VD, VI, Buff} <: AutoDiff.AbstractHessianProd
    model::Model
    func::Func
    map::VI
    stack::NetworkStack{VT, VD}
    ∂stack::NetworkStack{VT, VD}
    t1sF::VD
    ∂t1sF::VD
    buffer::Buff
end

function HessianProd(polar::PolarForm{T, VI, VT, MT}, func::AutoDiff.AbstractExpression, map::Vector{Int}) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    n_cons = length(func)

    nmap = length(map)
    map_device = map |> VI

    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    VD = A{t1s{1}}

    stack = NetworkStack(nbus, ngen, nlines, VT, VD)
    init!(polar, stack)

    ∂stack = NetworkStack(nbus, ngen, nlines, VT, VD)

    t1sF = zeros(Float64, n_cons) |> VD
    adj_t1sF = similar(t1sF)

    intermediate = nothing
    return HessianProd(
        polar, func, map_device, stack, ∂stack, t1sF, adj_t1sF,
        intermediate,
    )
end

function hprod!(
    H::HessianProd, hv, stack, λ, v,
)
    @assert length(hv) == length(v)

    # Init dual variables
    H.stack.input .= stack.input
    empty!(H.∂stack)
    H.∂t1sF .= λ

    # Seeding
    AutoDiff.seed!(H, v)
    # Forward
    H.func(H.t1sF, H.stack)
    # Forward-over-Reverse
    adjoint!(H.func, H.∂stack, H.stack, H.∂t1sF)

    AutoDiff.getpartials_kernel!(hv, H)
    return
end

function _hessian_sparsity(polar::PolarForm, func)
    m = length(func)
    nbus = get(polar, PS.NumberOfBuses())
    Vre = Float64[i for i in 1:nbus]
    Vim = Float64[i for i in nbus+1:2*nbus]
    V = Vre .+ im .* Vim
    y = rand(m)
    return matpower_hessian(polar, func, V, y)
end

struct FullHessian{Model, Func, Stack, VD, SMT, VI} <: AutoDiff.AbstractFullHessian
    model::Model
    func::Func
    map::VI
    stack::Stack
    ∂stack::Stack
    coloring::VI
    ncolors::Int
    t1sF::VD
    ∂t1sF::VD
    H::SMT
end

function _get_hessian_colors(polar::PolarForm, func::AutoDiff.AbstractExpression, map::Vector{Int})
    H = _hessian_sparsity(polar, func)::SparseMatrixCSC
    Hsub = H[map, map] # reorder
    colors = AutoDiff.SparseDiffTools.matrix_colors(Hsub)
    return (Hsub, colors)
end

function FullHessian(polar::PolarForm{T, VI, VT, MT}, func::AutoDiff.AbstractExpression, map::Vector{Int}) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    n_cons = length(func)

    nmap = length(map)
    map_device = map |> VI

    H_host, coloring = _get_hessian_colors(polar, func, map)
    ncolors = length(unique(coloring))
    VD = A{ForwardDiff.Dual{Nothing, Float64, ncolors}}

    H = H_host |> SMT

    # Structures
    stack = NetworkStack(nbus, ngen, nlines, VT, VD)
    init!(polar, stack)

    ∂stack = NetworkStack(nbus, ngen, nlines, VT, VD)
    t1sF = zeros(Float64, n_cons) |> VD
    adj_t1sF = similar(t1sF)

    coloring = coloring |> VI

    hess = FullHessian(
        polar, func, map_device, stack, ∂stack, coloring, ncolors, t1sF, adj_t1sF,
        H,
    )

    # seed
    AutoDiff.seed_coloring!(hess, coloring)

    return hess
end

function hessian!(
    H::FullHessian, stack, λ,
)
    # init
    AutoDiff.set_value!(H, stack.input)
    empty!(H.∂stack)
    H.∂t1sF .= λ
    # forward pass
    H.func(H.t1sF, H.stack)
    # forward-over-reverse pass
    adjoint!(H.func, H.∂stack, H.stack, H.∂t1sF)
    # extract partials
    AutoDiff.partials!(H)
    return H.H
end

function BatchHessian(
    polar::PolarForm{T, VI, VT, MT},
    func::AutoDiff.AbstractExpression,
    map::Vector{Int},
    blk_map::Vector{Int},
    k::Int,
) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    n_cons = length(func) * k

    nmap = length(map)

    H_host, coloring = _get_hessian_colors(polar, func, map)
    ncolors = length(unique(coloring))
    VD = A{ForwardDiff.Dual{Nothing, Float64, ncolors}}

    H = repeat(H_host, k) |> SMT

    # Structures
    stack = BlockNetworkStack(k, nbus, ngen, nlines, VT, VD)
    init!(polar, stack)

    ∂stack = BlockNetworkStack(k, nbus, ngen, nlines, VT, VD)
    t1sF = zeros(Float64, n_cons) |> VD
    adj_t1sF = similar(t1sF)

    coloring = repeat(coloring, k) |> VI

    map_device = blk_map |> VI
    hess = FullHessian(
        polar, func, map_device, stack, ∂stack, coloring, ncolors, t1sF, adj_t1sF,
        H,
    )

    # seed
    AutoDiff.seed_coloring!(hess, coloring)

    return hess
end
