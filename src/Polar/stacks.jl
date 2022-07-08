
abstract type AbstractNetworkStack{VT} <: AutoDiff.AbstractStack{VT} end

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

"""
    NetworkStack{VT,VD,MT} <: AbstractNetworkStack{VT}
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

### Examples

```jldoctest; setup=:(using ExaPF)
julia> polar = ExaPF.load_polar("case9");

julia> stack = ExaPF.NetworkStack(polar)
21-elements NetworkStack{Vector{Float64}}

julia> stack.vmag
9-element Vector{Float64}:
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
    for f in fieldnames(typeof(intermediate))
        fill!(getfield(intermediate, f), 0.0)
    end

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

nblocks(stack::NetworkStack) = 1

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


"""
    BlockNetworkStack{MT,MD,MI} <: AbstractStack{MT}

Store the variables of the `N` different scenarios
associated to the polar formulation. Extend [`NetworkStack`](@ref).

The variables are stored in the field `input`, and
are ordered as follows
```
    input = [ vmag^1, ..., vmag^N, vang^1, ..., vang^N, pgen^1, ..., pgen^N]
```
---
    BlockNetworkStack(polar::PolarForm, k::Int)

Create a `BlockNetworkStack` with `k` different scenarios using the data stored inside `polar`.

---
    BlockNetworkStack(polar::PolarForm, pload::Array, qload::Array)
Create a `BlockNetworkStack` using the load scenarios stored
inside the 2-dimensional arrays `pload` and `qload`.

---
    BlockNetworkStack(k::Int, nbus::Int, ngen::Int, nlines::Int, VT::Type)
Create an empty `BlockNetworkStack` with the size needed to stored `k` different scenarios.

"""
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
    for f in fieldnames(typeof(intermediate))
        fill!(getfield(intermediate, f), 0.0)
    end

    # Parameters: loads
    params = VT(undef, 2*k*nbus) ; fill!(params, 0.0)
    p0 = pointer(params)
    pload = unsafe_wrap(VT, p0, k*nbus)
    p1 = pointer(params, k*nbus+1)
    qload = unsafe_wrap(VT, p1, k*nbus)

    return BlockNetworkStack(k, input, vmag, vang, pgen, ψ, intermediate, params, pload, qload)
end
function BlockNetworkStack(polar::BlockPolarForm{T,VI,VT,MT}) where {T,VI,VT,MT}
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    nlines = get(polar, PS.NumberOfLines())
    stack = BlockNetworkStack(nblocks(polar), nbus, ngen, nlines, VT, VT)
    init!(polar, stack)
    return stack
end
function BlockNetworkStack(
    polar::BlockPolarForm,
    ploads::Array{Float64, 2},
    qloads::Array{Float64, 2},
)
    @assert size(ploads) == size(qloads)
    k = size(ploads, 2)
    blk_stack = BlockNetworkStack(polar)

    copyto!(blk_stack.pload, ploads)
    copyto!(blk_stack.qload, qloads)
    return blk_stack
end


function Base.show(io::IO, stack::BlockNetworkStack)
    print(io, "$(length(stack.input))-elements BlockNetworkStack{$(typeof(stack.input))}")
end

nblocks(stack::BlockNetworkStack) = stack.k

function init!(polar::BlockPolarForm, stack::BlockNetworkStack; loads=true)
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

        if loads
            i = (s - 1) * nload  +1
            copyto!(stack.pload, i, pload, 1, nload)
            copyto!(stack.qload, i, qload, 1, nload)
        end
    end
end

function Base.empty!(stack::BlockNetworkStack)
    fill!(stack.vmag, 0.0)
    fill!(stack.vang, 0.0)
    fill!(stack.pgen, 0.0)
    fill!(stack.ψ, 0.0)
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

function blockcopy!(stack::BlockNetworkStack, map::AbstractArray, x::AbstractArray)
    nx = length(x)
    @assert length(map) % nx == 0
    nb = div(length(map), nx)
    for k in 1:nb, i in eachindex(x)
        stack.input[map[i + (k-1)*nx]] = x[i]
    end
end

