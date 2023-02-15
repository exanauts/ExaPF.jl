
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
    vuser::VD # custom variables defined by user
    # INTERMEDIATE
    ψ::VD    # nonlinear basis ψ(vmag, vang)
    intermediate::NT
    # Parameters
    params::VT
    pload::VT
    qload::VT
    nblocks::Int
end

function NetworkStack(nbus::Int, ngen::Int, nlines::Int, nuser::Int, k::Int, VT::Type, VD::Type)
    m = (2 * nbus + ngen + nuser) * k
    input = VD(undef, m) ; fill!(input, 0.0)
    # Wrap directly array x to avoid dealing with views
    p0 = pointer(input)
    vmag = unsafe_wrap(VD, p0, k * nbus)
    p1 = pointer(input, k*nbus+1)
    vang = unsafe_wrap(VD, p1, k * nbus)
    p2 = pointer(input, 2*k*nbus+1)
    pgen = unsafe_wrap(VD, p2, k * ngen)
    p3 = pointer(input, 2*k*nbus+k*ngen+1)
    vuser = unsafe_wrap(VD, p3, k * nuser)

    # Basis function
    ψ = VD(undef, k*(2*nlines + nbus)) ; fill!(ψ, 0.0)
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

    return NetworkStack(input, vmag, vang, pgen, vuser, ψ, intermediate, params, pload, qload, k)
end
function NetworkStack(nbus::Int, ngen::Int, nlines::Int, ncustoms::Int, VT::Type, VD::Type)
    return NetworkStack(nbus, ngen, nlines, ncustoms, 1, VT, VD)
end

function NetworkStack(polar::AbstractPolarFormulation{T,VI,VT,MT}) where {T,VI,VT,MT}
    nbus = get(polar, PS.NumberOfBuses())
    ngen = get(polar, PS.NumberOfGenerators())
    nlines = get(polar, PS.NumberOfLines())
    stack = NetworkStack(nbus, ngen, nlines, polar.ncustoms, nblocks(polar), VT, VT)
    init!(polar, stack)
    return stack
end

function Base.show(io::IO, stack::NetworkStack)
    print(io, "$(length(stack.input))-elements NetworkStack{$(typeof(stack.input))}")
end

nblocks(stack::NetworkStack) = stack.nblocks

"""
    init!(polar::PolarForm, stack::NetworkStack)

Set `stack.input` with the initial values specified
in the base [`PS.PowerNetwork`](@ref) object.

"""
function init!(polar::AbstractPolarFormulation, stack::NetworkStack; update_loads=true)
    vmag = get(polar.network, PS.VoltageMagnitude())
    vang = get(polar.network, PS.VoltageAngle())
    pgen = get(polar.network, PS.ActivePower())
    pload = get(polar.network, PS.ActiveLoad())
    qload = get(polar.network, PS.ReactiveLoad())

    nbus = length(vmag)
    ngen = length(pgen)
    nload = length(pload)

    for s in 1:nblocks(stack)
        i = (s - 1) * nbus  + 1
        copyto!(stack.vmag, i, vmag, 1, nbus)
        copyto!(stack.vang, i, vang, 1, nbus)

        i = (s - 1) * ngen  + 1
        copyto!(stack.pgen, i, pgen, 1, ngen)

        if update_loads
            i = (s - 1) * nload  + 1
            copyto!(stack.pload, i, pload, 1, nload)
            copyto!(stack.qload, i, qload, 1, nload)
        end
    end
end

function Base.empty!(stack::NetworkStack)
    fill!(stack.input, 0.0)
    fill!(stack.ψ, 0.0)
    return
end

function bounds(polar::AbstractPolarFormulation{T, VI, VT, MT}, stack::NetworkStack) where {T, VI, VT, MT}
    nbus = polar.network.nbus
    vmag_min, vmag_max = PS.bounds(polar.network, PS.Buses(), PS.VoltageMagnitude())
    vang_min, vang_max = fill(-Inf, nbus), fill(Inf, nbus)
    pgen_min, pgen_max = PS.bounds(polar.network, PS.Generators(), PS.ActivePower())

    lb = [
        repeat(vmag_min, nblocks(polar));
        repeat(vang_min, nblocks(polar));
        repeat(pgen_min, nblocks(polar));
    ]
    ub = [
        repeat(vmag_max, nblocks(polar));
        repeat(vang_max, nblocks(polar));
        repeat(pgen_max, nblocks(polar));
    ]
    return convert(VT, lb), convert(VT, ub)
end

"Get complex voltage from `NetworkStack`."
voltage(buf::NetworkStack) = buf.vmag .* exp.(im .* buf.vang)
voltage_host(buf::NetworkStack) = voltage(buf) |> Array

function set_params!(stack::NetworkStack, pload::Array, qload::Array)
    copyto!(stack.pload, pload[:])
    copyto!(stack.qload, qload[:])
end

function blockcopy!(stack::NetworkStack, map::AbstractArray, x::AbstractArray)
    nx = length(x)
    @assert length(map) % nx == 0
    nb = div(length(map), nx)
    for k in 1:nb, i in eachindex(x)
        stack.input[map[i + (k-1)*nx]] = x[i]
    end
end

