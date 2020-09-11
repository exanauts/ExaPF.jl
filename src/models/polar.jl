# Polar formulation
#
struct PolarForm{T, IT, VT, AT} <: AbstractFormulation where {T, IT, VT, AT}
    network::PS.PowerNetwork
    device::Device
    x_min::VT
    x_max::VT
    u_min::VT
    u_max::VT
    # costs
    costs_coefficients::AT
    # Constants
    active_load::VT
    reactive_load::VT
    indexing::IndexingCache{IT}
    # struct
    ybus_re::Spmat{IT, VT}
    ybus_im::Spmat{IT, VT}
    AT::Type
end

function PolarForm(pf::PS.PowerNetwork, device; nocost=false)
    if isa(device, CPU)
        IT = Vector{Int}
        VT = Vector
        M = SparseMatrixCSC
        AT = Array
    elseif isa(device, CUDADevice)
        IT = CuArray{Int64, 1, Nothing}
        VT = CuVector
        M = CuSparseMatrixCSR
        AT = CuArray
    end

    npv = PS.get(pf, PS.NumberOfPVBuses())
    npq = PS.get(pf, PS.NumberOfPQBuses())
    nref = PS.get(pf, PS.NumberOfSlackBuses())
    ngens = PS.get(pf, PS.NumberOfGenerators())

    ybus_re, ybus_im = Spmat{IT, VT{Float64}}(pf.Ybus)
    # Get coefficients penalizing the generation of the generators
    coefs = convert(AT{Float64, 2}, PS.get_costs_coefficients(pf))
    # Move load to the target device
    pload , qload = real.(pf.sload), imag.(pf.sload)

    # Move the indexing to the target device
    idx_gen = convert(VT{Int}, PS.get(pf, PS.GeneratorIndexes()))
    idx_ref = convert(VT{Int}, PS.get(pf, PS.SlackIndexes()))
    idx_pv = convert(VT{Int}, PS.get(pf, PS.PVIndexes()))
    idx_pq = convert(VT{Int}, PS.get(pf, PS.PQIndexes()))

    # Bounds
    ## Get bounds on active power
    p_min, p_max = PS.bounds(pf, PS.Generator(), PS.ActivePower())
    ## Get bounds on voltage magnitude
    v_min, v_max = PS.bounds(pf, PS.Buses(), PS.VoltageMagnitude())
    ## Instantiate arrays
    nᵤ = nref + 2*npv
    nₓ = npv + 2*npq
    u_min = fill(-Inf, nᵤ)
    u_max = fill( Inf, nᵤ)
    x_min = fill(-Inf, nₓ)
    x_max = fill( Inf, nₓ)
    ## Bounds on v_pq
    x_min[npv+npq+1:end] .= v_min[idx_pq]
    x_max[npv+npq+1:end] .= v_max[idx_pq]
    ## Bounds on v_pv
    u_min[nref+npv+1:end] .= v_min[idx_pv]
    u_max[nref+npv+1:end] .= v_max[idx_pv]
    ## Bounds on v_ref
    u_min[1:nref] .= v_min[idx_ref]
    u_max[1:nref] .= v_max[idx_ref]
    ## Bounds on p_pv
    for i in 1:ngens
        bus = idx_gen[i]
        i_pv = findfirst(isequal(bus), idx_pv)
        if !isnothing(i_pv)
            u_min[i_pv + nref] = p_min[i]
            u_max[i_pv + nref] = p_max[i]
        end
    end

    indexing = IndexingCache(idx_pv, idx_pq, idx_ref, idx_gen)

    return PolarForm{Float64, IT, VT{Float64}, AT{Float64,  2}}(
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

function initial(form::PolarForm{T, IT, VT, AT}, v::AbstractVariable) where {T, IT, VT, AT}
    pbus = real.(form.network.sbus) |> VT
    qbus = imag.(form.network.sbus) |> VT
    vmag = abs.(form.network.vbus) |> VT
    vang = angle.(form.network.vbus) |> VT
    return get(form, v, vmag, vang, pbus, qbus)
end

function get(
    polar::PolarForm{T, IT, VT, AT},
    ::State,
    vmag::VT,
    vang::VT,
    pbus::VT,
    qbus::VT,
) where {T, IT, VT, AT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    # build vector x
    dimension = get(polar, NumberOfState())
    x = VT(undef, dimension)
    x[1:npv] = vang[polar.network.pv]
    x[npv+1:npv+npq] = vang[polar.network.pq]
    x[npv+npq+1:end] = vmag[polar.network.pq]

    return x
end

function get(
    polar::PolarForm{T, IT, VT, AT},
    ::Control,
    vmag::VT,
    vang::VT,
    pbus::VT,
    qbus::VT,
) where {T, IT, VT, AT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    pload = polar.active_load
    # build vector u
    dimension = get(polar, NumberOfControl())
    u = VT(undef, dimension)
    u[1:nref] = vmag[polar.network.ref]
    # u is equal to active power of generator (Pᵍ)
    # As P = Pᵍ - Pˡ , we get
    u[nref + 1:nref + npv] = pbus[polar.network.pv] + pload[polar.network.pv]
    u[nref + npv + 1:nref + 2*npv] = vmag[polar.network.pv]
    return u
end

function get(
    polar::PolarForm{T, IT, VT, AT},
    ::Parameters,
    vmag::VT,
    vang::VT,
    pbus::VT,
    qbus::VT,
) where {T, IT, VT, AT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    # build vector p
    dimension = nref + 2*npq
    p = VT(undef, dimension)
    p[1:nref] = vang[polar.network.ref]
    p[nref + 1:nref + npq] = pbus[polar.network.pq]
    p[nref + npq + 1:nref + 2*npq] = qbus[polar.network.pq]
    return p
end

# Bridge with buses' attributes
function get(polar::PolarForm{T, IT, VT, AT}, ::PS.Buses, ::PS.VoltageMagnitude, x, u, p; V=eltype(x)) where {T, IT, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    MT = polar.AT
    vmag = MT{V, 1}(undef, nbus)
    vmag[polar.network.pq] = x[npq+npv+1:end]
    vmag[polar.network.ref] = u[1:nref]
    vmag[polar.network.pv] = u[nref + npv + 1:nref + 2*npv]
    return vmag
end
function get(polar::PolarForm{T, IT, VT, AT}, ::PS.Buses, ::PS.VoltageAngle, x, u, p; V=eltype(x)) where {T, IT, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    MT = polar.AT
    vang = MT{V, 1}(undef, nbus)
    vang[polar.network.pq] = x[npv+1:npv+npq]
    vang[polar.network.pv] = x[1:npv]
    vang[polar.network.ref] = p[1:nref]
    return vang
end
function get(polar::PolarForm{T, IT, VT, AT}, ::PS.Buses, ::PS.ActivePower, x, u, p; V=eltype(x)) where {T, IT, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    vmag = get(polar, PS.Buses(), PS.VoltageMagnitude(), x, u, p)
    vang = get(polar, PS.Buses(), PS.VoltageAngle(), x, u, p)
    MT = polar.AT
    pinj = MT{V, 1}(undef, nbus)
    pinj[polar.network.pv] = u[nref + 1:nref + npv] - polar.active_load[polar.network.pv]
    pinj[polar.network.pq] = p[nref + 1:nref + npq]
    for bus in polar.network.ref
        pinj[bus] = PS.get_power_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
    end
    return pinj
end
function get(polar::PolarForm{T, IT, VT, AT}, ::PS.Buses, ::PS.ReactivePower, x, u, p; V=eltype(x)) where {T, IT, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    vmag = get(polar, PS.Buses(), PS.VoltageMagnitude(), x, u, p)
    vang = get(polar, PS.Buses(), PS.VoltageAngle(), x, u, p)
    qinj = VT(undef, nbus)
    qinj[polar.network.pq] = p[nref + npq + 1:nref + 2*npq]
    for bus in [polar.network.ref; polar.network.pv]
        qinj[bus] = PS.get_react_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
    end
    return qinj
end

# Bridge with generators' attributes
function get(polar::PolarForm{T, IT, VT, AT}, ::PS.Generator, ::PS.ActivePower, x, u, p; V=eltype(x)) where {T, IT, VT, AT}
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())

    index_ref = polar.indexing.index_ref
    index_pv = polar.indexing.index_pv
    index_gen = polar.indexing.index_generators

    # Get voltages.
    vmag = get(polar, PS.Buses(), PS.VoltageMagnitude(), x, u, p; V=V)
    vang = get(polar, PS.Buses(), PS.VoltageAngle(), x, u, p; V=V)

    MT = polar.AT
    pg = MT{V, 1}(undef, ngen)
    # TODO: check the complexity of this for loop
    for i in 1:ngen
        bus = index_gen[i]
        if bus in index_ref
            inj = PS.get_power_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
            pg[i] = inj + polar.active_load[bus]
        else
            ipv = findfirst(isequal(bus), index_pv)
            pg[i] = u[nref + ipv]
        end
    end

    return pg
end

function bounds(polar::PolarForm{T, IT, VT, AT}, ::State) where {T, IT, VT, AT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    fr_ = npq + npv + 1
    to_ = 2*npq + npv
    return polar.x_min[fr_:to_], polar.x_max[fr_:to_]
end
function bounds(polar::PolarForm{T, IT, VT, AT}, ::Control) where {T, IT, VT, AT}
    return polar.u_min, polar.u_max
end

function get_network_state(polar::PolarForm{T, IT, VT, AT}, x, u, p; V=Float64) where {T, IT, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    pf = polar.network

    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq

    MT = polar.AT
    vmag = MT{V, 1}(undef, nbus)
    vang = MT{V, 1}(undef, nbus)
    pinj = MT{V, 1}(undef, nbus)
    qinj = MT{V, 1}(undef, nbus)

    vang[pv] .= x[1:npv]
    vang[pq] .= x[npv+1:npv+npq]
    vmag[pq] .= x[npv+npq+1:end]

    vmag[ref] .= u[1:nref]
    pinj[pv] .= u[nref + 1:nref + npv] - polar.active_load[pv]
    vmag[pv] .= u[nref + npv + 1:nref + 2*npv]

    vang[ref] .= p[1:nref]
    pinj[pq] .= p[nref + 1:nref + npq]
    qinj[pq] .= p[nref + npq + 1:nref + 2*npq]

    for bus in ref
        pinj[bus] = PS.get_power_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
        qinj[bus] = PS.get_react_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
    end

    for bus in pv
        qinj[bus] = PS.get_react_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
    end

    return vmag, vang, pinj, qinj
end

function load!(network::NetworkState, x, u, p, polar::PolarForm{T, IT, VT, AT}) where {T, IT, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())

    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq

    network.vmag[pq] .= x[1:npq]
    network.vang[pq] .= x[npq + 1:2*npq]
    network.vang[pv] .= x[2*npq + 1:2*npq + npv]

    network.vmag[ref] .= u[1:nref]
    network.pinj[pv] .= u[nref + 1:nref + npv] - polar.active_load[pv]
    network.vmag[pv] .= u[nref + npv + 1:nref + 2*npv]

    network.vang[ref] .= p[1:nref]
    network.pinj[pq] .= p[nref + 1:nref + npq]
    network.qinj[pq] .= p[nref + npq + 1:nref + 2*npq]

    for bus in ref
        network.pinj[bus] = PS.get_power_injection(bus, network.vmag, network.vang, polar.ybus_re, polar.ybus_im)
        network.qinj[bus] = PS.get_react_injection(bus, network.vmag, network.vang, polar.ybus_re, polar.ybus_im)
    end

    for bus in pv
        network.qinj[bus] = PS.get_react_injection(bus, network.vmag, network.vang, polar.ybus_re, polar.ybus_im)
    end
end

function power_balance(polar::PolarForm, x, u, p; V=Float64)
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    # Indexing
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq

    # Network state
    Vm, Va, pbus, qbus = get_network_state(polar, x, u, p; V=V)
    F = similar(x)
    fill!(F, 0.0)
    residualFunction_polar!(F, Vm, Va,
                            polar.ybus_re, polar.ybus_im,
                            pbus, qbus, pv, pq, nbus)
    return F
end

# TODO: find better naming
function init_ad_factory(polar::PolarForm{T, IT, VT, AT}, x, u, p) where {T, IT, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    n_states = get(polar, NumberOfState())
    # Indexing
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    # Network state
    Vm, Va, pbus, qbus = get_network_state(polar, x, u, p)
    F = VT(undef, n_states)
    fill!(F, zero(T))
    # Build the AD Jacobian structure
    stateJacobianAD = AD.StateJacobianAD(F, Vm, Va,
                                         polar.ybus_re, polar.ybus_im, pbus, qbus, pv, pq, ref, nbus)
    designJacobianAD = AD.DesignJacobianAD(F, Vm, Va,
                                           polar.ybus_re, polar.ybus_im, pbus, qbus, pv, pq, ref, nbus)
    return stateJacobianAD, designJacobianAD
end

function jacobian(polar::PolarForm, jac::AD.AbstractJacobianAD, x, u, p)
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    # Indexing
    ref = polar.indexing.index_ref
    pv = polar.indexing.index_pv
    pq = polar.indexing.index_pq
    # Network state
    Vm, Va, pbus, qbus = get_network_state(polar, x, u, p)
    AD.residualJacobianAD!(jac, residualFunction_polar!, Vm, Va,
                           polar.ybus_re, polar.ybus_im, pbus, qbus, pv, pq, ref, nbus, TIMER)
    return jac.J
end

function powerflow(
    polar::PolarForm{T, IT, VT, AT},
    jacobian::AD.StateJacobianAD,
    x::VT,
    u::VT,
    p::VT;
    kwargs...
) where {T, IT, VT, AT}
    Vm, Va, pbus, qbus = get_network_state(polar, x, u, p)
    network = NetworkState{VT}(Vm, Va, pbus, qbus, VT(undef, 0), VT(undef, 0))
    return powerflow(polar, jacobian, network; kwargs...)
end

function powerflow(
    polar::PolarForm{T, IT, VT, AT},
    jacobian::AD.StateJacobianAD,
    network::NetworkState{VT};
    npartitions=2,
    solver="default",
    preconditioner=Precondition.NoPreconditioner(),
    tol=1e-7,
    maxiter=20,
    verbose_level=0,
) where {T, IT, VT, AT}
    # Retrieve parameter and initial voltage guess
    Vm, Va, pbus, qbus = network.vmag, network.vang, network.pinj, network.qinj

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
    F = VT(undef, n_states)
    dx = similar(F)
    fill!(F, zero(T))
    fill!(dx, zero(T))

    # Evaluate residual function
    residualFunction_polar!(F, Vm, Va,
                            polar.ybus_re, polar.ybus_im,
                            pbus, qbus, pv, pq, nbus)

    # check for convergence
    normF = norm(F, Inf)
    if verbose_level >= VERBOSE_LEVEL_HIGH
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
        n_iters = Iterative.ldiv!(dx, J, F, solver, preconditioner, TIMER)
        push!(linsol_iters, n_iters)
        # Sometimes it is better to move backward
        dx .= -dx

        # update voltage
        @timeit TIMER "Update voltage" begin
            if (npv != 0)
                # Va[pv] .= Va[pv] .+ dx[j5:j6]
                Vapv .= Vapv .+ dx56
            end
            if (npq != 0)
                # Vm[pq] .= Vm[pq] .+ dx[j1:j2]
                Vmpq .= Vmpq .+ dx12
                # Va[pq] .= Va[pq] .+ dx[j3:j4]
                Vapq .= Vapq .+ dx34
            end
        end

        fill!(F, zero(T))
        @timeit TIMER "Residual function" begin
            residualFunction_polar!(F, Vm, Va,
                polar.ybus_re, polar.ybus_im,
                pbus, qbus, pv, pq, nbus)
        end

        @timeit TIMER "Norm" normF = norm(F, Inf)
        if verbose_level >= VERBOSE_LEVEL_HIGH
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

    xk = get(polar, State(), Vm, Va, pbus, qbus)

    # Timer outputs display
    if verbose_level >= VERBOSE_LEVEL_MEDIUM
        show(TIMER)
        println("")
    end
    reset_timer!(TIMER)
    conv = ConvergenceStatus(converged, iter, normF, sum(linsol_iters))
    return xk, conv
end

# Cost function
function cost_production(polar::PolarForm, x, u, p; V=Float64)
    # TODO: this getter is particularly inefficient on GPU
    power_generations = get(polar, PS.Generator(), PS.ActivePower(), x, u, p; V=V)
    c0 = polar.costs_coefficients[:, 2]
    c1 = polar.costs_coefficients[:, 3]
    c2 = polar.costs_coefficients[:, 4]
    # Return quadratic cost
    cost = sum(c0 .+ c1 .* power_generations + c2 .* power_generations.^2)
    return cost
end

# Generic inequality constraints
# We add constraint only on vmag_pq
function state_constraint(polar::PolarForm, g, x, u, p; V=Float64)
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    g .= x[npv+npq+1:end]
    return
end
size_constraint(polar::PolarForm{T, IT, VT, AT}, ::typeof(state_constraint)) where {T, IT, VT, AT} = PS.get(polar.network, PS.NumberOfPQBuses())
bounds(polar::PolarForm, ::typeof(state_constraint)) = bounds(polar, State())

# Here, the power constraints are ordered as:
# g = [P_ref; Q_ref; Q_pv]
function power_constraints(polar::PolarForm, g, x, u, p; V=Float64)
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    Vm, Va, pbus, qbus = get_network_state(polar, x, u, p; V=V)
    ref = convert(polar.AT{Int, 1}, polar.network.ref)
    pv = convert(polar.AT{Int, 1}, polar.network.pv)

    cnt = 1
    # Constraint on P_ref (generator) (P_inj = P_g - P_load)
    for bus in ref
        g[cnt] = PS.get_power_injection(bus, Vm, Va, polar.ybus_re, polar.ybus_im) + polar.active_load[bus]
        cnt += 1
    end
    # Constraint on Q_ref (generator) (Q_inj = Q_g - Q_load)
    for bus in ref
        g[cnt] = PS.get_react_injection(bus, Vm, Va, polar.ybus_re, polar.ybus_im) + polar.reactive_load[bus]
        cnt += 1
    end
    # Constraint on Q_pv (generator) (Q_inj = Q_g - Q_load)
    for bus in pv
        g[cnt] = PS.get_react_injection(bus, Vm, Va, polar.ybus_re, polar.ybus_im) + polar.reactive_load[bus]
        cnt += 1
    end
    return
end
function size_constraint(polar::PolarForm{T, IT, VT, AT}, ::typeof(power_constraints)) where {T, IT, VT, AT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    return 2*nref + npv
end
function bounds(polar::PolarForm{T, IT, VT, AT}, ::typeof(power_constraints)) where {T, IT, VT, AT}
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())

    # Get all bounds (lengths of p_min, p_max, q_min, q_max equal to ngen)
    p_min, p_max = PS.bounds(polar.network, PS.Generator(), PS.ActivePower())
    q_min, q_max = PS.bounds(polar.network, PS.Generator(), PS.ReactivePower())

    index_gen = PS.get(polar.network, PS.GeneratorIndexes())
    index_pv = polar.network.pv
    index_ref = polar.network.ref

    MT = polar.AT
    pq_min = MT{T, 1}(undef, 2*nref + npv)
    pq_max = MT{T, 1}(undef, 2*nref + npv)
    # TODO: check the complexity of this for loop
    # Remind that the ordering is
    # g = [P_ref; Q_ref; Q_pv]
    for i in 1:ngen
        bus = index_gen[i]
        # First, try to find if index bus is a slack bus
        # (most efficient to test index_ref first, as most of the time
        #  index_ref has length equal to 1)
        i_ref = findfirst(isequal(bus), index_ref)
        if !isnothing(i_ref)
            # fill P_ref
            pq_min[i_ref] = p_min[i]
            pq_max[i_ref] = p_max[i]
            # fill Q_ref
            pq_min[i_ref + nref] = q_min[i]
            pq_max[i_ref + nref] = q_max[i]
        else # is a PV bus
            i_pv = findfirst(isequal(bus), index_pv)
            # fill Q_pv
            pq_min[i_pv + 2*nref] = q_min[i]
            pq_max[i_pv + 2*nref] = q_max[i]
        end
    end
    return pq_min, pq_max
end
