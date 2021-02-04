# Polar formulation
#

struct StateJacobianStructure{IT} <: AbstractJacobianStructure where {IT}
    sparsity::Function
    map::IT
end

struct ControlJacobianStructure{IT} <: AbstractJacobianStructure where {IT}
    sparsity::Function
    map::IT
end

struct PolarForm{T, IT, VT, AT} <: AbstractFormulation where {T, IT, VT, AT}
    network::PS.PowerNetwork
    device::KA.Device
    # bounds
    x_min::VT
    x_max::VT
    u_min::VT
    u_max::VT
    # costs
    costs_coefficients::AT
    # Constant loads
    active_load::VT
    reactive_load::VT
    # Indexing of the PV, PQ and slack buses
    indexing::IndexingCache{IT}
    # struct
    topology::NetworkTopology{IT, VT}
    AT::Type
    # Jacobian structures and indexing
    statejacobian::StateJacobianStructure
    controljacobian::ControlJacobianStructure
end

include("kernels.jl")
include("gradients.jl")
include("hessians.jl")
include("getters.jl")
include("adjoints.jl")
include("constraints.jl")

function PolarForm(pf::PS.PowerNetwork, device)
    if isa(device, KA.CPU)
        IT = Vector{Int}
        VT = Vector{Float64}
        M = SparseMatrixCSC
        AT = Array
    elseif isa(device, KA.CUDADevice)
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
    # Move load to the target device
    pload = convert(VT, PS.get(pf, PS.ActiveLoad()))
    qload = convert(VT, PS.get(pf, PS.ReactiveLoad()))

    # Move the indexing to the target device
    idx_gen = PS.get(pf, PS.GeneratorIndexes())
    idx_ref = PS.get(pf, PS.SlackIndexes())
    idx_pv = PS.get(pf, PS.PVIndexes())
    idx_pq = PS.get(pf, PS.PQIndexes())
    # Build-up reverse index for performance
    pv_to_gen = similar(idx_pv)
    ref_to_gen = similar(idx_ref)
    ## We assume here that the indexing of generators is the same
    ## as in MATPOWER
    for i in 1:ngens
        bus = idx_gen[i]
        i_pv = findfirst(isequal(bus), idx_pv)
        if !isnothing(i_pv)
            pv_to_gen[i_pv] = i
        else
            i_ref = findfirst(isequal(bus), idx_ref)
            if !isnothing(i_ref)
                ref_to_gen[i_ref] = i
            end
        end
    end

    gidx_gen = convert(IT, idx_gen)
    gidx_ref = convert(IT, idx_ref)
    gidx_pv = convert(IT, idx_pv)
    gidx_pq = convert(IT, idx_pq)
    gref_to_gen = convert(IT, ref_to_gen)
    gpv_to_gen = convert(IT, pv_to_gen)

    # Bounds
    ## Get bounds on active power
    p_min, p_max = PS.bounds(pf, PS.Generator(), PS.ActivePower())
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
    state_jacobian_structure = StateJacobianStructure(residual_jacobian, IT(statemap))

    controlmap = vcat(idx_ref, idx_pv, idx_pv .+ nbus)
    control_jacobian_structure = ControlJacobianStructure{IT}(residual_jacobian, IT(controlmap))

    return PolarForm{Float64, IT, VT, AT{Float64,  2}}(
        pf, device,
        x_min, x_max, u_min, u_max,
        coefs, pload, qload,
        indexing,
        topology,
        AT,
        state_jacobian_structure, control_jacobian_structure
    )
end

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

function get(polar::PolarForm, attr::PS.AbstractNetworkAttribute)
    return get(polar.network, attr)
end

# Setters
function setvalues!(polar::PolarForm, ::PS.ActiveLoad, values)
    @assert length(polar.active_load) == length(values)
    copyto!(polar.active_load, values)
end
function setvalues!(polar::PolarForm, ::PS.ReactiveLoad, values)
    @assert length(polar.reactive_load) == length(values)
    copyto!(polar.reactive_load, values)
end

## Bounds
function bounds(polar::PolarForm{T, IT, VT, AT}, ::State) where {T, IT, VT, AT}
    return polar.x_min, polar.x_max
end
function bounds(polar::PolarForm{T, IT, VT, AT}, ::Control) where {T, IT, VT, AT}
    return polar.u_min, polar.u_max
end

function initial(form::PolarForm{T, IT, VT, AT}, v::AbstractVariable) where {T, IT, VT, AT}
    pbus = real.(form.network.sbus) |> VT
    qbus = imag.(form.network.sbus) |> VT
    vmag = abs.(form.network.vbus) |> VT
    vang = angle.(form.network.vbus) |> VT
    return get(form, v, vmag, vang, pbus, qbus)
end

function get(form::PolarForm{T, IT, VT, AT}, ::PhysicalState) where {T, IT, VT, AT}
    nbus = PS.get(form.network, PS.NumberOfBuses())
    ngen = PS.get(form.network, PS.NumberOfGenerators())
    n_state = get(form, NumberOfState())
    gen2bus = form.indexing.index_generators
    return PolarNetworkState{VT}(nbus, ngen, n_state, gen2bus)
end

function init_buffer!(form::PolarForm{T, IT, VT, AT}, buffer::PolarNetworkState) where {T, IT, VT, AT}
    # FIXME
    pbus = real.(form.network.sbus)
    qbus = imag.(form.network.sbus)
    vmag = abs.(form.network.vbus)
    vang = angle.(form.network.vbus)

    pg = get(form.network, PS.ActivePower())
    qg = get(form.network, PS.ReactivePower())

    copyto!(buffer.vmag, vmag)
    copyto!(buffer.vang, vang)
    copyto!(buffer.pg, pg)
    copyto!(buffer.qg, qg)
    copyto!(buffer.pinj, pbus)
    copyto!(buffer.qinj, qbus)
end

function power_balance!(polar::PolarForm, buffer::PolarNetworkState)
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    # Indexing
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    Vm, Va, pbus, qbus = buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    F = buffer.balance
    fill!(F, 0.0)
    residual_polar!(
        F, Vm, Va,
        ybus_re, ybus_im,
        pbus, qbus, pv, pq, nbus
    )
end

# TODO: find better naming
function init_autodiff_factory(polar::PolarForm{T, IT, VT, AT}, buffer::PolarNetworkState) where {T, IT, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())
    nₓ = get(polar, NumberOfState())
    nᵤ = get(polar, NumberOfControl())
    # Take indexing on the CPU as we initiate AutoDiff on the CPU
    ref = polar.network.ref
    pv = polar.network.pv
    pq = polar.network.pq
    # Network state
    Vm, Va, pbus, qbus = buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj
    F = buffer.balance
    fill!(F, zero(T))
    # Build the AutoDiff Jacobian structure
    statejacobian = AutoDiff.Jacobian(polar.statejacobian, F, Vm, Va,
        ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus,
        AutoDiff.StateJacobian()
    )
    controljacobian = AutoDiff.Jacobian(polar.controljacobian, F, Vm, Va,
        ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus,
        AutoDiff.ControlJacobian()
    )

    # Build the AutoDiff structure for the objective
    ∇fₓ = xzeros(VT, nₓ)
    ∇fᵤ = xzeros(VT, nᵤ)
    adjoint_pg = similar(buffer.pg)
    adjoint_vm = similar(Vm)
    adjoint_va = similar(Va)
    # Build cache for Jacobian vector-product
    jvₓ = xzeros(VT, nₓ)
    jvᵤ = xzeros(VT, nᵤ)
    adjoint_flow = xzeros(VT, 2 * nbus)
    objectiveAD = AdjointStackObjective(∇fₓ, ∇fᵤ, adjoint_pg, adjoint_vm, adjoint_va, jvₓ, jvᵤ, adjoint_flow)
    return statejacobian, controljacobian, objectiveAD
end

function jacobian(polar::PolarForm, jac::AutoDiff.Jacobian, buffer::PolarNetworkState, jac_type::AutoDiff.AbstractJacobian)
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())
    # Indexing
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    # Network state
    Vm, Va, pbus, qbus = buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj
    AutoDiff.residual_jacobian!(jac, residual_polar!, Vm, Va,
                           ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus, jac_type)
    return jac.J
end

function powerflow(
    polar::PolarForm{T, IT, VT, AT},
    jacobian::AutoDiff.Jacobian,
    buffer::PolarNetworkState{IT,VT},
    algo::NewtonRaphson;
    solver=DirectSolver(),
) where {T, IT, VT, AT}
    # Retrieve parameter and initial voltage guess
    Vm, Va, pbus, qbus = buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj

    nbus = PS.get(polar.network, PS.NumberOfBuses())
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    n_states = get(polar, NumberOfState())
    nvbus = length(polar.network.vbus)
    ybus_re, ybus_im = get(polar.topology, PS.BusAdmittanceMatrix())

    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq

    # iteration variables
    iter = 0
    converged = false

    # indices
    j1 = 1
    j2 = npv
    j3 = j2 + 1
    j4 = j2 + npq
    j5 = j4 + 1
    j6 = j4 + npq

    # form residual function directly on target device
    F = buffer.balance
    dx = buffer.dx
    fill!(F, zero(T))
    fill!(dx, zero(T))

    # Evaluate residual function
    residual_polar!(
        F, Vm, Va, ybus_re, ybus_im,
        pbus, qbus, pv, pq, nbus
    )

    # check for convergence
    normF = norm(F, Inf)
    if algo.verbose >= VERBOSE_LEVEL_LOW
        @printf("Iteration %d. Residual norm: %g.\n", iter, normF)
    end
    if normF < algo.tol
        converged = true
    end

    linsol_iters = Int[]
    Vapv = view(Va, pv)
    Vapq = view(Va, pq)
    Vmpq = view(Vm, pq)
    dx12 = view(dx, j5:j6) # Vmqp
    dx34 = view(dx, j3:j4) # Vapq
    dx56 = view(dx, j1:j2) # Vapv

    @timeit TIMER "Newton" while ((!converged) && (iter < algo.maxiter))

        iter += 1

        @timeit TIMER "Jacobian" begin
            AutoDiff.residual_jacobian!(jacobian, residual_polar!,
                                   Vm, Va,
                                   ybus_re, ybus_im, pbus, qbus, pv, pq, ref, nbus, AutoDiff.StateJacobian())
        end
        J = jacobian.J

        # Find descent direction
        if isa(solver, LinearSolvers.AbstractIterativeLinearSolver)
            @timeit TIMER "Preconditioner" LinearSolvers.update!(solver, J)
        end
        @timeit TIMER "Linear Solver" n_iters = LinearSolvers.ldiv!(solver, dx, J, F)
        push!(linsol_iters, n_iters)

        # update voltage
        @timeit TIMER "Update voltage" begin
            # Sometimes it is better to move backward
            if (npv != 0)
                # Va[pv] .= Va[pv] .+ dx[j5:j6]
                Vapv .= Vapv .- dx56
            end
            if (npq != 0)
                # Vm[pq] .= Vm[pq] .+ dx[j1:j2]
                Vmpq .= Vmpq .- dx12
                # Va[pq] .= Va[pq] .+ dx[j3:j4]
                Vapq .= Vapq .- dx34
            end
        end

        fill!(F, zero(T))
        @timeit TIMER "Residual function" begin
            residual_polar!(F, Vm, Va,
                ybus_re, ybus_im,
                pbus, qbus, pv, pq, nbus)
        end

        @timeit TIMER "Norm" normF = xnorm(F)
        if algo.verbose >= VERBOSE_LEVEL_LOW
            @printf("Iteration %d. Residual norm: %g.\n", iter, normF)
        end

        if normF < algo.tol
            converged = true
        end
    end

    if algo.verbose >= VERBOSE_LEVEL_HIGH
        if converged
            @printf("N-R converged in %d iterations.\n", iter)
        else
            @printf("N-R did not converge.\n")
        end
    end

    # Timer outputs display
    if algo.verbose >= VERBOSE_LEVEL_MEDIUM
        show(TIMER)
        println("")
    end
    return ConvergenceStatus(converged, iter, normF, sum(linsol_iters))
end

# Cost function
function cost_production(polar::PolarForm, x, u, p)
    # TODO: this getter is particularly inefficient on GPU
    power_generations = get(polar, PS.Generator(), PS.ActivePower(), x, u, p)
    return cost_production(polar, power_generations)
end
# TODO: write up a function more efficient in GPU
function cost_production(polar::PolarForm, pg)
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    coefs = polar.costs_coefficients
    c2 = @view coefs[:, 2]
    c3 = @view coefs[:, 3]
    c4 = @view coefs[:, 4]
    # Return quadratic cost
    # NB: this operation induces three allocations on the GPU,
    #     but is faster than writing the sum manually
    cost = sum(c2 .+ c3 .* pg .+ c4 .* pg.^2)
    return cost
end

# Utils
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

