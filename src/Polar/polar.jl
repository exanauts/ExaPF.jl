# Polar formulation
#
#
#
include("caches.jl")

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
    # bounds
    x_min::VT
    x_max::VT
    u_min::VT
    u_max::VT
    # costs
    costs_coefficients::MT
    # Indexing of the PV, PQ and slack buses
    indexing::IndexingCache{IT}
    # struct
    topology::NetworkTopology{IT, VT}
    # Jacobian indexing
    mapx::IT
    mapu::IT
    # Hessian structures and indexing
    hessianstructure::HessianStructure
end

include("kernels.jl")
include("derivatives.jl")
include("Constraints/constraints.jl")
include("powerflow.jl")
include("objective.jl")
include("batch.jl")

function PolarForm(pf::PS.PowerNetwork, device::KA.Device)
    if isa(device, KA.CPU)
        IT = Vector{Int}
        VT = Vector{Float64}
        M = SparseMatrixCSC
        AT = Array
    elseif isa(device, KA.GPU)
        IT = CUDA.CuVector{Int64}
        VT = CUDA.CuVector{Float64}
        M = CUSPARSE.CuSparseMatrixCSR
        AT = CUDA.CuArray
    end

    nbus = PS.get(pf, PS.NumberOfBuses())
    npv = PS.get(pf, PS.NumberOfPVBuses())
    npq = PS.get(pf, PS.NumberOfPQBuses())
    nref = PS.get(pf, PS.NumberOfSlackBuses())
    ngens = PS.get(pf, PS.NumberOfGenerators())

    topology = NetworkTopology{IT, VT}(pf)
    # Get coefficients penalizing the generation of the generators
    coefs = convert(AT{Float64, 2}, PS.get_costs_coefficients(pf))

    # Move the indexing to the target device
    idx_gen = PS.get(pf, PS.GeneratorIndexes())
    idx_ref = PS.get(pf, PS.SlackIndexes())
    idx_pv = PS.get(pf, PS.PVIndexes())
    idx_pq = PS.get(pf, PS.PQIndexes())
    # Build-up reverse index for performance
    pv_to_gen = PS.get(pf, PS.PVToGeneratorsIndex())
    ref_to_gen = PS.get(pf, PS.SlackToGeneratorsIndex())

    gidx_gen = convert(IT, idx_gen)
    gidx_ref = convert(IT, idx_ref)
    gidx_pv = convert(IT, idx_pv)
    gidx_pq = convert(IT, idx_pq)
    gref_to_gen = convert(IT, ref_to_gen)
    gpv_to_gen = convert(IT, pv_to_gen)

    # Bounds
    ## Get bounds on active power
    p_min, p_max = PS.bounds(pf, PS.Generators(), PS.ActivePower())
    p_min = convert(VT, p_min)
    p_max = convert(VT, p_max)
    ## Get bounds on voltage magnitude
    v_min, v_max = PS.bounds(pf, PS.Buses(), PS.VoltageMagnitude())
    v_min = convert(VT, v_min)
    v_max = convert(VT, v_max)
    ## Instantiate arrays
    nᵤ = nref + 2*npv
    nₓ = npv + 2*npq
    u_min = convert(VT, fill(-Inf, nᵤ))
    u_max = convert(VT, fill( Inf, nᵤ))
    x_min = convert(VT, fill(-Inf, nₓ))
    x_max = convert(VT, fill( Inf, nₓ))
    ## Bounds on v_pq
    x_min[npv+npq+1:end] .= v_min[gidx_pq]
    x_max[npv+npq+1:end] .= v_max[gidx_pq]
    ## Bounds on v_pv
    u_min[nref+1:nref+npv] .= v_min[gidx_pv]
    u_max[nref+1:nref+npv] .= v_max[gidx_pv]
    ## Bounds on v_ref
    u_min[1:nref] .= v_min[gidx_ref]
    u_max[1:nref] .= v_max[gidx_ref]
    ## Bounds on p_pv
    u_min[nref+npv+1:nref+2*npv] .= p_min[gpv_to_gen]
    u_max[nref+npv+1:nref+2*npv] .= p_max[gpv_to_gen]

    indexing = IndexingCache(gidx_pv, gidx_pq, gidx_ref, gidx_gen, gpv_to_gen, gref_to_gen)
    mappv = [i + nbus for i in idx_pv]
    mappq = [i + nbus for i in idx_pq]
    # Ordering for x is (θ_pv, θ_pq, v_pq)
    statemap = vcat(mappv, mappq, idx_pq)
    controlmap = vcat(idx_ref, idx_pv, idx_pv .+ nbus)
    hessianmap = vcat(statemap, idx_ref, idx_pv, idx_pv .+ 2*nbus)
    hessianstructure = HessianStructure(IT(hessianmap))
    return PolarForm{Float64, IT, VT, AT{Float64,  2}}(
        pf, device,
        x_min, x_max, u_min, u_max,
        coefs,
        indexing,
        topology,
        statemap, controlmap,
        hessianstructure
    )
end
# Convenient constructor
PolarForm(datafile::String, device) = PolarForm(PS.PowerNetwork(datafile), device)

array_type(polar::PolarForm) = array_type(polar.device)

# Getters
function get(polar::PolarForm, ::NumberOfState)
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    return 2*npq + npv
end

function get(polar::PolarForm, ::NumberOfControl)
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    return nref + 2*npv
end

get(polar::PolarForm, attr::PS.AbstractNetworkAttribute) = get(polar.network, attr)

index_buses_host(polar) = PS.get(polar.network, PS.AllBusesIndex())
index_buses_device(polar) = index_buses(polar.indexing)

index_generators_host(polar) = PS.get(polar.network, PS.AllGeneratorsIndex())
index_generators_device(polar) = index_generators(polar.indexing)

## Bounds
function bounds(polar::PolarForm{T, IT, VT, MT}, ::State) where {T, IT, VT, MT}
    return polar.x_min, polar.x_max
end

function bounds(polar::PolarForm{T, IT, VT, MT}, ::Control) where {T, IT, VT, MT}
    return polar.u_min, polar.u_max
end

# Initial position
function initial(polar::PolarForm{T, IT, VT, MT}, X::Union{State,Control}) where {T, IT, VT, MT}
    ref, pv, pq = index_buses_host(polar)
    _, _, pv2gen = index_generators_host(polar)
    # Load data from PowerNetwork
    vmag = abs.(polar.network.vbus)
    vang = angle.(polar.network.vbus)
    pg = get(polar.network, PS.ActivePower())

    if isa(X, State)
        # build vector x
        return [vang[pv] ; vang[pq] ; vmag[pq]] |> VT
    elseif isa(X, Control)
        return [vmag[ref] ; vmag[pv] ; pg[pv2gen]] |> VT
    end
end

function get(form::PolarForm{T, IT, VT, MT}, ::PhysicalState) where {T, IT, VT, MT}
    nbus = PS.get(form.network, PS.NumberOfBuses())
    ngen = PS.get(form.network, PS.NumberOfGenerators())
    n_state = get(form, NumberOfState())
    gen2bus = form.indexing.index_generators
    return PolarNetworkState{VT}(nbus, ngen, n_state, gen2bus)
end

function get!(
    polar::PolarForm{T, IT, VT, MT},
    ::State,
    x::AbstractVector,
    buffer::PolarNetworkState
) where {T, IT, VT, MT}
    npv = get(polar, PS.NumberOfPVBuses())
    npq = get(polar, PS.NumberOfPQBuses())
    nref = get(polar, PS.NumberOfSlackBuses())
    ref, pv, pq = index_buses_host(polar)
    # Copy values of vang and vmag into x
    # NB: this leads to 3 memory allocation on the GPU
    #     we use indexing on the CPU, as for some reason
    #     we get better performance than with the indexing on the GPU
    #     stored in the buffer polar.indexing.
    x[1:npv] .= @view buffer.vang[pv]
    x[npv+1:npv+npq] .= @view buffer.vang[pq]
    x[npv+npq+1:npv+2*npq] .= @view buffer.vmag[pq]
end

function get!(
    polar::PolarForm{T, IT, VT, MT},
    ::Control,
    u::AbstractVector,
    buffer::PolarNetworkState,
) where {T, IT, VT, MT}
    npv = get(polar, PS.NumberOfPVBuses())
    npq = get(polar, PS.NumberOfPQBuses())
    nref = get(polar, PS.NumberOfSlackBuses())
    ref, pv, pq = index_buses_host(polar)
    _, _, pv2gen = index_generators_host(polar)
    # build vector u
    nᵤ = get(polar, NumberOfControl())
    u[1:nref] .= @view buffer.vmag[ref]
    u[nref + 1:nref + npv] .= @view buffer.vmag[pv]
    u[nref + npv + 1:nref + 2*npv] .= @view buffer.pgen[pv2gen]
    return u
end

function init_buffer!(form::PolarForm{T, IT, VT, MT}, buffer::PolarNetworkState) where {T, IT, VT, MT}
    # FIXME: add proper getters in PowerSystem
    vmag = abs.(form.network.vbus)
    vang = angle.(form.network.vbus)
    pd = PS.get(form.network, PS.ActiveLoad())
    qd = PS.get(form.network, PS.ReactiveLoad())

    pg = get(form.network, PS.ActivePower())
    qg = get(form.network, PS.ReactivePower())

    copyto!(buffer.vmag, vmag)
    copyto!(buffer.vang, vang)
    copyto!(buffer.pgen, pg)
    copyto!(buffer.qgen, qg)
    copyto!(buffer.pload, pd)
    copyto!(buffer.qload, qd)

    fill!(buffer.pnet, 0.0)
    fill!(buffer.qnet, 0.0)
    copyto!(view(buffer.pnet, form.indexing.index_generators), pg)
    copyto!(view(buffer.qnet, form.indexing.index_generators), qg)
    return
end

function direct_linear_solver(polar::PolarForm)
    is_cpu = isa(polar.device, KA.CPU)
    if is_cpu
        jac = jacobian_sparsity(polar, power_balance, State())
        return LinearSolvers.DirectSolver(jac)
    else
        # Factorization is not yet supported on the GPU
        return LinearSolvers.DirectSolver(nothing)
    end
end

function build_preconditioner(polar::PolarForm; nblocks=-1)
    jac = jacobian_sparsity(polar, power_balance, State())
    n = size(jac, 1)
    npartitions = if nblocks > 0
        nblocks
    else
        div(n, 32)
    end
    return LinearSolvers.BlockJacobiPreconditioner(jac, npartitions, polar.device)
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
    n_states = get(polar, NumberOfState())
    n_controls = get(polar, NumberOfControl())
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

