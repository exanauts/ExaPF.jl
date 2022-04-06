#=
    Ordering of input:
    [ vmag^1, ..., vmag^N, vang^1, ..., vang^N, pgen^1, ..., pgen^N]
=#
struct BlockNetworkStack{MT,MD,NT} <: AbstractNetworkStack{MT}
    k::Int
    # INPUT
    input::MD
    vmag::MD # voltage magnitudes
    vang::MD # voltage angles
    pgen::MD # active power generations
    # INTERMEDIATE
    ψ::MD    # nonlinear basis ψ(vmag, vang)
    intermediate::NT
    # Parameters
    params::MT
    pload::MT
    qload::MT
end

function BlockNetworkStack(k, nbus, ngen, nlines, VT, VD)
    m = (2*nbus + ngen) * k
    input = VD(undef, m) ; fill!(input, 0.0)
    # Wrap directly array x to avoid dealing with views
    p0 = pointer(input)
    vmag = unsafe_wrap(VD, p0, k*nbus)
    p1 = pointer(input, k*nbus+1)
    vang = unsafe_wrap(VD, p1, k*nbus)
    p2 = pointer(input, 2*k*nbus+1)
    pgen = unsafe_wrap(VD, p2, k*ngen)

    # Basis function
    ψ = VD(undef, k * (2*nlines+nbus)) ; fill!(ψ, 0.0)
    # Intermediate expressions to avoid unecessary allocations
    intermediate = (
        c = VD(undef, k*ngen),     # buffer for costs
        sfp = VD(undef, k*nlines), # buffer for line-flow
        sfq = VD(undef, k*nlines), # buffer for line-flow
        stp = VD(undef, k*nlines), # buffer for line-flow
        stq = VD(undef, k*nlines), # buffer for line-flow
        ∂edge_vm_fr = VD(undef, k*nlines), # buffer for basis
        ∂edge_vm_to = VD(undef, k*nlines), # buffer for basis
        ∂edge_va_fr = VD(undef, k*nlines), # buffer for basis
        ∂edge_va_to = VD(undef, k*nlines), # buffer for basis
    )

    # Parameters: loads
    params = VT(undef, 2*k*nbus) ; fill!(params, 0.0)
    p0 = pointer(params)
    pload = unsafe_wrap(VT, p0, k*nbus)
    p1 = pointer(params, k*nbus+1)
    qload = unsafe_wrap(VT, p1, k*nbus)

    return BlockNetworkStack(k, input, vmag, vang, pgen, ψ, intermediate, params, pload, qload)
end

function Base.show(io::IO, stack::BlockNetworkStack)
    print(io, "$(length(stack.input))-elements BlockNetworkStack{$(typeof(stack.input))}")
end

nbatches(stack::BlockNetworkStack) = stack.k

function init!(polar::PolarForm, stack::BlockNetworkStack)
    vmag = get(polar.network, PS.VoltageMagnitude())
    vang = get(polar.network, PS.VoltageAngle())
    pgen = get(polar.network, PS.ActivePower())
    pload = get(polar.network, PS.ActiveLoad())
    qload = get(polar.network, PS.ReactiveLoad())

    nbus = length(vmag)
    ngen = length(pgen)
    nload = length(pload)

    for s in 1:stack.k
        i = (s - 1) * nbus  +1
        copyto!(stack.vmag, i, vmag, 1, nbus)
        copyto!(stack.vang, i, vang, 1, nbus)

        i = (s - 1) * ngen  +1
        copyto!(stack.pgen, i, pgen, 1, ngen)

        i = (s - 1) * nload  +1
        copyto!(stack.pload, i, pload, 1, nload)
        copyto!(stack.qload, i, qload, 1, nload)
    end
end

function BlockNetworkStack(polar::PolarForm{T,VI,VT,MT}, k::Int) where {T,VI,VT,MT}
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    nlines = get(polar, PS.NumberOfLines())
    stack = BlockNetworkStack(k, nbus, ngen, nlines, VT, VT)
    init!(polar, stack)
    return stack
end

function Base.empty!(stack::BlockNetworkStack)
    fill!(stack.vmag, 0.0)
    fill!(stack.vang, 0.0)
    fill!(stack.pgen, 0.0)
    fill!(stack.ψ, 0.0)
    fill!(stack.pload, 0.0)
    fill!(stack.qload, 0.0)
    return
end

function bounds(polar::PolarForm{T, VI, VT, MT}, stack::BlockNetworkStack) where {T, VI, VT, MT}
    k = stack.k
    nbus = polar.network.nbus
    vmag_min, vmag_max = PS.bounds(polar.network, PS.Buses(), PS.VoltageMagnitude())
    vang_min, vang_max = fill(-Inf, nbus), fill(Inf, nbus)
    pgen_min, pgen_max = PS.bounds(polar.network, PS.Generators(), PS.ActivePower())

    lb = [
        repeat(vmag_min, k);
        repeat(vang_min, k);
        repeat(pgen_min, k);
    ]
    ub = [
        repeat(vmag_max, k);
        repeat(vang_max, k);
        repeat(pgen_max, k);
    ]
    return convert(VT, lb), convert(VT, ub)
end
