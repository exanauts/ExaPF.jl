
struct MyHessian{Model, Func, VD, VI, T1, T2, Buff} <: AutoDiff.AbstractHessian
    model::Model
    func::Func
    map::VI
    state::NetworkStack{VD}
    ∂state::NetworkStack{VD}
    host_t1sseeds::T1 # Needed because seeds have to be created on the host
    t1sseeds::T2
    t1sF::VD
    ∂t1sF::VD
    buffer::Buff
end

function MyHessian(polar::PolarForm{T, VI, VT, MT}, func::AbstractExpression, map::Vector{Int}) where {T, VI, VT, MT}
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

    intermediate = _get_intermediate_stack(polar, network_basis, VD, 1)
    return MyHessian(
        polar, func, map_device, stack, ∂stack, host_t1sseeds, t1sseeds, t1sF, adj_t1sF,
        intermediate,
    )
end

function hprod!(
    H::MyHessian, hv, state, λ, v,
)
    @assert length(hv) == length(v)

    # Init dual variables
    H.state.input .= state.input
    empty!(H.∂state)
    H.∂t1sF .= λ

    # Seeding
    nmap = length(H.map)
    # Init seed
    _init_seed_hessian!(H.t1sseeds, H.host_t1sseeds, v, nmap)
    myseed!(H.state, state, H.t1sseeds, H.map, H.model.device)
    forward_eval_intermediate(H.model, H.state)
    H.func(H.t1sF, H.state)

    # Reverse
    adjoint!(H.func, H.∂state, H.state, H.∂t1sF)
    reverse_eval_intermediate(H.model, H.∂state, H.state, H.buffer)

    AutoDiff.getpartials_kernel!(hv, H.∂state.input, H.map, H.model.device)
    return
end

