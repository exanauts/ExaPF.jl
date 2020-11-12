# Polar formulation
#
struct PolarForm{T, IT, VT, AT} <: AbstractFormulation where {T, IT, VT, AT}
    network::PS.PowerNetwork
    device::Device
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
    ybus_re::Spmat{IT, VT}
    ybus_im::Spmat{IT, VT}
    AT::Type
end

include("kernels.jl")
include("getters.jl")
include("adjoints.jl")
include("constraints.jl")

function PolarForm(pf::PS.PowerNetwork, device; nocost=false)
    if isa(device, CPU)
        IT = Vector{Int}
        VT = Vector{Float64}
        M = SparseMatrixCSC
        AT = Array
    elseif isa(device, CUDADevice)
        IT = CuVector{Int64}
        VT = CuVector{Float64}
        M = CuSparseMatrixCSR
        AT = CuArray
    end

    npv = PS.get(pf, PS.NumberOfPVBuses())
    npq = PS.get(pf, PS.NumberOfPQBuses())
    nref = PS.get(pf, PS.NumberOfSlackBuses())
    ngens = PS.get(pf, PS.NumberOfGenerators())

    ybus_re, ybus_im = Spmat{IT, VT}(pf.Ybus)
    # Get coefficients penalizing the generation of the generators
    coefs = convert(AT{Float64, 2}, PS.get_costs_coefficients(pf))
    # Move load to the target device
    pload = convert(VT, real.(pf.sload))
    qload = convert(VT, imag.(pf.sload))

    # Move the indexing to the target device
    idx_gen = PS.get(pf, PS.GeneratorIndexes())
    idx_ref = PS.get(pf, PS.SlackIndexes())
    idx_pv = PS.get(pf, PS.PVIndexes())
    idx_pq = PS.get(pf, PS.PQIndexes())
    # Build-up reverse index for performance
    pv_to_gen = similar(idx_pv)
    ref_to_gen = similar(idx_ref)
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
    u_min[nref+npv+1:end] .= v_min[gidx_pv]
    u_max[nref+npv+1:end] .= v_max[gidx_pv]
    ## Bounds on v_ref
    u_min[1:nref] .= v_min[gidx_ref]
    u_max[1:nref] .= v_max[gidx_ref]
    ## Bounds on p_pv
    u_min[nref+1:nref+npv] .= p_min[gpv_to_gen]
    u_max[nref+1:nref+npv] .= p_max[gpv_to_gen]

    indexing = IndexingCache(gidx_pv, gidx_pq, gidx_ref, gidx_gen, gpv_to_gen, gref_to_gen)

    return PolarForm{Float64, IT, VT, AT{Float64,  2}}(
        pf, device,
        x_min, x_max, u_min, u_max,
        coefs, pload, qload,
        indexing,
        ybus_re, ybus_im,
        AT,
    )
end

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
    pbus = real.(form.network.sbus) |> VT
    qbus = imag.(form.network.sbus) |> VT
    vmag = abs.(form.network.vbus) |> VT
    vang = angle.(form.network.vbus) |> VT
    ngen = PS.get(form.network, PS.NumberOfGenerators())
    pg = xzeros(VT, ngen)
    qg = xzeros(VT, ngen)

    npv = PS.get(form.network, PS.NumberOfPVBuses())
    npq = PS.get(form.network, PS.NumberOfPQBuses())
    balance = xzeros(VT, 2*npq+npv)
    dx = xzeros(VT, 2*npq+npv)
    return PolarNetworkState{VT}(vmag, vang, pbus, qbus, pg, qg, balance, dx)
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

    F = buffer.balance
    fill!(F, 0.0)
    residualFunction_polar!(F, Vm, Va,
                            polar.ybus_re, polar.ybus_im,
                            pbus, qbus, pv, pq, nbus)
end

# TODO: find better naming
function init_ad_factory(polar::PolarForm{T, IT, VT, AT}, buffer::PolarNetworkState) where {T, IT, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nₓ = get(polar, NumberOfState())
    nᵤ = get(polar, NumberOfControl())
    # Take indexing on the CPU as we initiate AD on the CPU
    ref = polar.network.ref
    pv = polar.network.pv
    pq = polar.network.pq
    # Network state
    Vm, Va, pbus, qbus = buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj
    F = buffer.balance
    fill!(F, zero(T))
    # Build the AD Jacobian structure
    stateJacobianAD = AD.StateJacobianAD(F, Vm, Va,
        polar.ybus_re, polar.ybus_im, pbus, qbus, pv, pq, ref, nbus
    )
    designJacobianAD = AD.DesignJacobianAD(F, Vm, Va,
        polar.ybus_re, polar.ybus_im, pbus, qbus, pv, pq, ref, nbus
    )

    # Build the AD structure for the objective
    ∇fₓ = xzeros(VT, nₓ)
    ∇fᵤ = xzeros(VT, nᵤ)
    adjoint_pg = similar(buffer.pg)
    adjoint_vm = similar(Vm)
    adjoint_va = similar(Va)
    # Build cache for Jacobian vector-product
    jvₓ = xzeros(VT, nₓ)
    jvᵤ = xzeros(VT, nᵤ)
    objectiveAD = AD.ObjectiveAD(∇fₓ, ∇fᵤ, adjoint_pg, adjoint_vm, adjoint_va, jvₓ, jvᵤ)
    return stateJacobianAD, designJacobianAD, objectiveAD
end

function jacobian(polar::PolarForm, jac::AD.AbstractJacobianAD, buffer::PolarNetworkState)
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    # Indexing
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    # Network state
    Vm, Va, pbus, qbus = buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj
    AD.residualJacobianAD!(jac, residualFunction_polar!, Vm, Va,
                           polar.ybus_re, polar.ybus_im, pbus, qbus, pv, pq, ref, nbus, TIMER)
    return jac.J
end

function powerflow(
    polar::PolarForm{T, IT, VT, AT},
    jacobian::AD.StateJacobianAD,
    buffer::PolarNetworkState{VT};
    solver=DirectSolver(),
    tol=1e-7,
    maxiter=20,
    verbose_level=0,
) where {T, IT, VT, AT}
    # Retrieve parameter and initial voltage guess
    Vm, Va, pbus, qbus = buffer.vmag, buffer.vang, buffer.pinj, buffer.qinj

    nbus = PS.get(polar.network, PS.NumberOfBuses())
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    n_states = get(polar, NumberOfState())

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
    residualFunction_polar!(F, Vm, Va,
                            polar.ybus_re, polar.ybus_im,
                            pbus, qbus, pv, pq, nbus)

    # check for convergence
    normF = norm(F, Inf)
    if verbose_level >= VERBOSE_LEVEL_LOW
        @printf("Iteration %d. Residual norm: %g.\n", iter, normF)
    end
    if normF < tol
        converged = true
    end

    linsol_iters = Int[]
    Vapv = view(Va, pv)
    Vapq = view(Va, pq)
    Vmpq = view(Vm, pq)
    dx12 = view(dx, j5:j6) # Vmqp
    dx34 = view(dx, j3:j4) # Vapq
    dx56 = view(dx, j1:j2) # Vapv

    @timeit TIMER "Newton" while ((!converged) && (iter < maxiter))

        iter += 1

        @timeit TIMER "Jacobian" begin
            AD.residualJacobianAD!(jacobian, residualFunction_polar!, Vm, Va,
                                   polar.ybus_re, polar.ybus_im, pbus, qbus, pv, pq, ref, nbus, TIMER)
        end
        J = jacobian.J

        # Find descent direction
        if isa(solver, LinearSolvers.AbstractIterativeLinearSolver)
            LinearSolvers.update!(solver, J)
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
            residualFunction_polar!(F, Vm, Va,
                polar.ybus_re, polar.ybus_im,
                pbus, qbus, pv, pq, nbus)
        end

        @timeit TIMER "Norm" normF = xnorm(F)
        if verbose_level >= VERBOSE_LEVEL_LOW
            @printf("Iteration %d. Residual norm: %g.\n", iter, normF)
        end

        if normF < tol
            converged = true
        end
    end

    if verbose_level >= VERBOSE_LEVEL_HIGH
        if converged
            @printf("N-R converged in %d iterations.\n", iter)
        else
            @printf("N-R did not converge.\n")
        end
    end

    # Timer outputs display
    if verbose_level >= VERBOSE_LEVEL_MEDIUM
        show(TIMER)
        println("")
    end
    conv = ConvergenceStatus(converged, iter, normF, sum(linsol_iters))
    return conv
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
