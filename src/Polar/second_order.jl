
struct HessianProd{Model, Func, VD, VI, T1, T2, Buff} <: AutoDiff.AbstractHessian
    model::Model
    func::Func
    map::VI
    stack::NetworkStack{VD}
    ∂stack::NetworkStack{VD}
    host_t1sseeds::T1 # Needed because seeds have to be created on the host
    t1sseeds::T2
    t1sF::VD
    ∂t1sF::VD
    buffer::Buff
end

function HessianProd(polar::PolarForm{T, VI, VT, MT}, func::AbstractExpression, map::Vector{Int}) where {T, VI, VT, MT}
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

    stack = NetworkStack(nbus, ngen, nlines, VD)
    ∂stack = NetworkStack(nbus, ngen, nlines, VD)

    t1sF = zeros(Float64, n_cons) |> VD
    adj_t1sF = similar(t1sF)

    # Seedings
    host_t1sseeds = Vector{ForwardDiff.Partials{1, Float64}}(undef, nmap)
    t1sseeds = A{ForwardDiff.Partials{1, Float64}}(undef, nmap)

    intermediate = nothing
    return HessianProd(
        polar, func, map_device, stack, ∂stack, host_t1sseeds, t1sseeds, t1sF, adj_t1sF,
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
    nmap = length(H.map)
    # Init seed
    AutoDiff.init_seed_hessian!(H.t1sseeds, H.host_t1sseeds, v, nmap)
    AutoDiff.seed!(H.stack, stack, H.t1sseeds, H.map, H.model.device)
    # Forward
    H.func(H.t1sF, H.stack)
    # Forward-over-Reverse
    adjoint!(H.func, H.∂stack, H.stack, H.∂t1sF)

    AutoDiff.getpartials_kernel!(hv, H.∂stack.input, H.map, H.model.device)
    return
end


struct FullHessian{Model, Func, VD, SMT, MT, VI, VP, Buff} <: AutoDiff.AbstractHessian
    model::Model
    func::Func
    map::VI
    stack::NetworkStack{VD}
    ∂stack::NetworkStack{VD}
    coloring::VI
    t1sseeds::VP
    t1sF::VD
    ∂t1sF::VD
    buffer::Buff
    compressedH::MT
    H::SMT
end

function get_hessian_colors(polar::PolarForm, func::AbstractExpression, map::Vector{Int})
    H = hessian_sparsity(polar, func)::SparseMatrixCSC
    Hsub = H[map, map] # reorder
    colors = AutoDiff.SparseDiffTools.matrix_colors(Hsub)
    return (Hsub, colors)
end

function FullHessian(polar::PolarForm{T, VI, VT, MT}, func::AbstractExpression, map::Vector{Int}) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    n_cons = length(func)

    nmap = length(map)
    map_device = map |> VI

    H_host, coloring = get_hessian_colors(polar, func, map)
    ncolor = length(unique(coloring))
    VD = A{ForwardDiff.Dual{Nothing, Float64, ncolor}}

    H = H_host |> SMT

    # Structures
    stack = NetworkStack(nbus, ngen, nlines, VD)
    ∂stack = NetworkStack(nbus, ngen, nlines, VD)
    t1sF = zeros(Float64, n_cons) |> VD
    adj_t1sF = similar(t1sF)

    # Seedings
    t1sseeds = AutoDiff.init_seed(coloring, ncolor, nmap) |> A

    compressedH = MT(undef, ncolor, nmap)
    coloring = coloring |> VI

    intermediate = nothing
    return FullHessian(
        polar, func, map_device, stack, ∂stack, coloring, t1sseeds, t1sF, adj_t1sF,
        intermediate, compressedH, H,
    )
end

function hessian!(
    H::FullHessian, stack, λ,
)
    # init
    H.stack.input .= stack.input
    empty!(H.∂stack)
    H.∂t1sF .= λ
    # seed
    AutoDiff.seed!(H.stack, stack, H.t1sseeds, H.map, H.model.device)
    # forward pass
    H.func(H.t1sF, H.stack)
    # forward-over-reverse pass
    adjoint!(H.func, H.∂stack, H.stack, H.∂t1sF)
    # uncompress
    AutoDiff.partials_hess!(H.compressedH, H.∂stack.input, H.map, H.model.device)
    AutoDiff.uncompress_kernel!(H.H, H.compressedH, H.coloring, H.model.device)
    return H.H
end

