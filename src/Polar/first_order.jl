function get_tape(polar::PolarForm, expr::AbstractExpression, ∂stack::NetworkStack{VT, Buf}) where {VT, Buf}
    # TODO
    intermediate = _get_intermediate_stack(polar, ExaPF.network_basis, VT, 1)
    return AutoDiff.TapeMemory(expr, ∂stack, intermediate)
end

function jacobian_transpose_product!(polar::PolarForm, pbm::AutoDiff.TapeMemory, jv, state, ∂v)
    ∂state = pbm.stack
    empty!(∂state)
    adjoint!(pbm.func, ∂state, state, ∂v)
    # Accumulate on vmag and vang
    reverse_eval_intermediate(polar, ∂state, state, pbm.intermediate)
    # Accumulate on x and u
    reverse_transfer!(
        polar, jv, ∂state,
    )
end

struct MyJacobian{Model, Func, VD, SMT, MT, VI, VP}
    model::Model
    func::Func
    map::VI
    stack::NetworkStack{VD}
    compressedJ::MT
    coloring::VI
    t1sseeds::VP
    t1sF::VD
    J::SMT
end

Base.size(jac::MyJacobian, n::Int) = size(jac.J, n)

# Ordering: [vmag, vang, pgen]

function my_map(polar::PolarForm, ::State)
    nbus = get(polar, PS.NumberOfBuses())
    ref, pv, pq = index_buses_device(polar)
    return Int[nbus .+ pv; nbus .+ pq; pq]
end
function my_map(polar::PolarForm, ::Control)
    nbus = get(polar, PS.NumberOfBuses())
    ref, pv, pq = index_buses_device(polar)
    pv2gen = polar.network.pv2gen
    return Int[ref; pv; 2*nbus .+ pv2gen]
end

number(polar::PolarForm, ::State) = get(polar, NumberOfState())
number(polar::PolarForm, ::Control) = get(polar, NumberOfControl())

# Coloring
function jacobian_sparsity(polar::PolarForm, func::AbstractExpression)
    nbus = get(polar, PS.NumberOfBuses())
    Vre = Float64[i for i in 1:nbus]
    Vim = Float64[i for i in nbus+1:2*nbus]
    V = Vre .+ im .* Vim
    return matpower_jacobian(polar, func, V)
end

function get_jacobian_colors(polar::PolarForm, func::AbstractExpression, map::Vector{Int})
    # Sparsity pattern
    J = jacobian_sparsity(polar, func)
    Jsub = J[:, map]
    # Coloring
    colors = AutoDiff.SparseDiffTools.matrix_colors(Jsub)
    return (Jsub, colors)
end

function MyJacobian(polar::PolarForm{T, VI, VT, MT}, func::AbstractExpression, map::Vector{Int}) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    n_cons = length(func)

    nmap = length(map)
    map_device = map |> VI

    J_host, coloring = get_jacobian_colors(polar, func, map)
    ncolor = size(unique(coloring),1)

    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    VD = A{t1s{ncolor}}

    J = J_host |> SMT

    # Seedings
    stack = NetworkStack(nbus, ngen, nlines, VD)
    t1sF = zeros(Float64, n_cons) |> VD
    t1sseeds = AutoDiff.init_seed(coloring, ncolor, nmap)

    # Move the seeds over to the device, if necessary
    gput1sseeds = A{ForwardDiff.Partials{ncolor,Float64}}(t1sseeds)
    compressedJ = MT(zeros(Float64, ncolor, n_cons))

    return MyJacobian(
        polar, func, map_device, stack, compressedJ, coloring, gput1sseeds, t1sF, J,
    )
end

@kernel function _seed_kernel!(
    duals::AbstractArray{ForwardDiff.Dual{T, V, N}}, @Const(x), @Const(seeds), @Const(map),
) where {T,V,N}
    i = @index(Global, Linear)
    duals[map[i]] = ForwardDiff.Dual{T,V,N}(x[map[i]], seeds[i])
end

function myseed!(dest, src, seeds, map, device)
    y = dest.input
    x = src.input
    ev = _seed_kernel!(device)(
        y, x, seeds, map, ndrange=length(map), dependencies=Event(device))
    wait(ev)
end

function jacobian!(
    jac::MyJacobian, state,
)
    # init
    jac.stack.input .= state.input
    jac.t1sF .= 0.0
    # seed
    myseed!(jac.stack, state, jac.t1sseeds, jac.map, jac.model.device)
    # forward pass
    forward_eval_intermediate(jac.model, jac.stack)
    jac.func(jac.t1sF, jac.stack)
    # uncompress
    AutoDiff.getpartials_kernel!(jac.compressedJ, jac.t1sF, jac.model.device)
    AutoDiff.uncompress_kernel!(jac.J, jac.compressedJ, jac.coloring, jac.model.device)
    return jac.J
end

