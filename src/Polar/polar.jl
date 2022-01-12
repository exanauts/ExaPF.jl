# Polar formulation

"""
    PolarForm{T, IT, VT, MT}

Takes as input a [`PS.PowerNetwork`](@ref) network and
implement the polar formulation model associated to this network.
The structure `PolarForm` stores the topology of the network, as
well as the complete indexing used in the polar formulation.

A `PolarForm` structure can be instantiated both on the host `CPU()`
or directly on the device `CUDADevice()`.
"""
struct PolarForm{T, IT, VT, MT} <: AbstractFormulation where {T, IT, VT, MT}
    network::PS.PowerNetwork
    device::KA.Device
end

include("functions.jl")
include("first_order.jl")
include("second_order.jl")
include("newton.jl")
include("legacy.jl")

function PolarForm(pf::PS.PowerNetwork, device::KA.Device)
    if isa(device, KA.CPU)
        IT = Vector{Int}
        VT = Vector{Float64}
        M = SparseMatrixCSC
        AT = Array
    elseif isa(device, KA.GPU)
        IT = CUDA.CuArray{Int64, 1, CUDA.Mem.DeviceBuffer}
        VT = CUDA.CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
        M = CUSPARSE.CuSparseMatrixCSR
        AT = CUDA.CuArray
    end

    return PolarForm{Float64, IT, VT, AT{Float64,  2}}(
        pf, device,
    )
end
# Convenient constructor
PolarForm(datafile::String, device) = PolarForm(PS.PowerNetwork(datafile), device)


# Ordering: [vmag, vang, pgen]

function my_map(polar::PolarForm, ::State)
    nbus = get(polar, PS.NumberOfBuses())
    ref, pv, pq = index_buses_host(polar)
    return Int[nbus .+ pv; nbus .+ pq; pq]
end
function my_map(polar::PolarForm, ::Control)
    nbus = get(polar, PS.NumberOfBuses())
    ref, pv, pq = index_buses_host(polar)
    pv2gen = polar.network.pv2gen
    return Int[ref; pv; 2*nbus .+ pv2gen]
end

number(polar::PolarForm, v::AbstractVariable) = length(my_map(polar, v))

# Getters
get(polar::PolarForm, attr::PS.AbstractNetworkAttribute) = get(polar.network, attr)

index_buses_host(polar) = PS.get(polar.network, PS.AllBusesIndex())
index_buses_device(polar) = index_buses(polar.indexing)

index_generators_host(polar) = PS.get(polar.network, PS.AllGeneratorsIndex())
index_generators_device(polar) = index_generators(polar.indexing)

# Power flow linear solvers
function powerflow_jacobian(polar)
    nbus = get(polar, PS.NumberOfBuses())
    v0 = polar.network.vbus .+ 0.01 .* rand(ComplexF64, nbus)
    return matpower_jacobian(polar, State(), power_balance, v0)
end

function powerflow_jacobian_device(polar)
    SpMT = default_sparse_matrix(polar.device)
    J = powerflow_jacobian(polar)
    return J |> SpMT
end

function Base.show(io::IO, polar::PolarForm)
    # Network characteristics
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    nlines = PS.get(polar.network, PS.NumberOfLines())
    # Polar formulation characteristics
    n_states = 2*npq + npv
    n_controls = nref + npv + ngen - 1
    print(io,   "Polar formulation model")
    println(io, " (instantiated on device $(polar.device))")
    println(io, "Network characteristics:")
    @printf(io, "    #buses:      %d  (#slack: %d  #PV: %d  #PQ: %d)\n", nbus, nref, npv, npq)
    println(io, "    #generators: ", ngen)
    println(io, "    #lines:      ", nlines)
    println(io, "giving a mathematical formulation with:")
    println(io, "    #controls:   ", n_controls)
    print(io,   "    #states  :   ", n_states)
end

