
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

struct MyJacobian{Func, VD, SMT, MT, VI, VP}
    func::Func
    stack::NetworkStack{VD}
    J::SMT
    compressedJ::MT
    coloring::VI
    map::VI
    t1sseeds::VP
    t1sF::VD
end


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

function MyJacobian(
    polar::PolarForm{T, VI, VT, MT}, func::AbstractExpression, variable,
) where {T, VI, VT, MT}
    (SMT, A) = get_jacobian_types(polar.device)

    pf = polar.network
    nbus = PS.get(pf, PS.NumberOfBuses())
    nlines = PS.get(pf, PS.NumberOfLines())
    ngen = PS.get(pf, PS.NumberOfGenerators())

    # Sparsity pattern
    J = jacobian_sparsity(polar, func, variable)
    # Coloring
    coloring = AutoDiff.SparseDiffTools.matrix_colors(J)
    ncolor = size(unique(coloring),1)

    m = size(J, 1)

    map = my_map(polar, variable)
    nmap = number(polar, variable)

    # Move Jacobian to the GPU
    J = convert(SMT, J)

    # Seedings
    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    stack = NetworkStack(nbus, ngen, nlines, A{t1s{ncolor}})
    t1sF = A{t1s{ncolor}}(zeros(Float64, m))
    t1sseeds = AutoDiff.init_seed(coloring, ncolor, nmap)

    # Move the seeds over to the device, if necessary
    gput1sseeds = A{ForwardDiff.Partials{ncolor,Float64}}(t1sseeds)
    compressedJ = MT(zeros(Float64, ncolor, m))

    return MyJacobian(
        func, stack, J, compressedJ, coloring, map, gput1sseeds, t1sF,
    )
end

@kernel function _seed_kernel2!(
    duals::AbstractArray{ForwardDiff.Dual{T, V, N}}, @Const(x),
    @Const(seeds), @Const(map),
) where {T,V,N}
    i = @index(Global, Linear)
    duals[map[i]] = ForwardDiff.Dual{T,V,N}(x[map[i]], seeds[i])
end

function myseed!(dest, src, seeds, map, device)
    y = dest.input
    x = src.input
    ev = _seed_kernel2!(device)(
        y, x, seeds, map, ndrange=length(map), dependencies=Event(device))
    wait(ev)
end

function jacobian!(
    polar::PolarForm, jac::MyJacobian, state,
)
    # init
    jac.stack.input .= state.input
    jac.t1sF .= 0.0
    # seed
    myseed!(jac.stack, state, jac.t1sseeds, jac.map, polar.device)
    # forward pass
    forward_eval_intermediate(polar, jac.stack)
    jac.func(jac.t1sF, jac.stack)
    # uncompress
    AutoDiff.getpartials_kernel!(jac.compressedJ, jac.t1sF, polar.device)
    AutoDiff.uncompress_kernel!(jac.J, jac.compressedJ, jac.coloring, polar.device)
    return jac.J
end

