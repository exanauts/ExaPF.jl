
const PS = PowerSystem
TIMER=  TimerOutput()

struct PolarForm{T, VT, AT} <: AbstractFormulation where {T, VT, AT}
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
    # struct
    ybus_re::ExaPF.Spmat
    ybus_im::ExaPF.Spmat
    AT::Type
end

function PolarForm(pf::PS.PowerNetwork, device)
    if isa(device, CPU)
        VT = Vector
        M = SparseMatrixCSC
        AT = Array
    elseif isa(device, CUDADevice)
        VT = CuVector
        M = CuSparseMatrixCSR
        AT = CuArray
    end
    ybus_re, ybus_im = ExaPF.Spmat{VT}(pf.Ybus)
    coefs = PS.get_costs_coefficients(pf) |> AT
    u_min, u_max, x_min, x_max, p_min, p_max = PS.get_bound_constraints(pf)
    pload , qload = real.(pf.sload), imag.(pf.sload)
    return PolarForm{Float64, VT{Float64}, AT{Float64,  2}}(
        pf, device,
        x_min, x_max, u_min, u_max,
        coefs, pload, qload,
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

function initial(form::PolarForm{T, VT, AT}, v::AbstractVariable) where {T, VT, AT}
    pbus = real.(form.network.sbus) |> VT
    qbus = imag.(form.network.sbus) |> VT
    vmag = abs.(form.network.vbus) |> VT
    vang = angle.(form.network.vbus) |> VT
    return get(form, v, vmag, vang, pbus, qbus)
end

function get(
    polar::PolarForm{T, VT, AT},
    ::State,
    vmag::VT,
    vang::VT,
    pbus::VT,
    qbus::VT,
) where {T<:Real, VT<:AbstractVector{T}, AT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    # build vector x
    dimension = 2*npq + npv
    x = VT(undef, dimension)
    x[1:npq] = vmag[polar.network.pq]
    x[npq + 1:2*npq] = vang[polar.network.pq]
    x[2*npq + 1:2*npq + npv] = vang[polar.network.pv]

    return x
end

function get(
    polar::PolarForm{T, VT, AT},
    ::Control,
    vmag::VT,
    vang::VT,
    pbus::VT,
    qbus::VT,
) where {T<:Real, VT<:AbstractVector{T}, AT}
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    pload = polar.active_load
    # build vector u
    dimension = 2*npv + nref
    u = VT(undef, dimension)
    u[1:nref] = vmag[polar.network.ref]
    # u is equal to active power of generator (Pᵍ)
    # As P = Pᵍ - Pˡ , we get
    u[nref + 1:nref + npv] = pbus[polar.network.pv] + pload[polar.network.pv]
    u[nref + npv + 1:nref + 2*npv] = vmag[polar.network.pv]
    return u
end

function get(
    polar::PolarForm{T, VT, AT},
    ::Parameters,
    vmag::VT,
    vang::VT,
    pbus::VT,
    qbus::VT,
) where {T<:Real, VT<:AbstractVector{T}, AT}
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
function get(polar::PolarForm{T, VT, AT}, ::PS.Buses, ::PS.VoltageMagnitude, x, u, p; V=eltype(x)) where {T, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    vmag = VT(undef, nbus)
    vmag[polar.network.pq] = x[1:npq]
    vmag[polar.network.ref] = u[1:nref]
    vmag[polar.network.pv] = u[nref + npv + 1:nref + 2*npv]
    return vmag
end
function get(polar::PolarForm{T, VT, AT}, ::PS.Buses, ::PS.VoltageAngle, x, u, p; V=eltype(x)) where {T, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    vang = VT(undef, nbus)
    vang[polar.network.pq] = x[npq + 1:2*npq]
    vang[polar.network.pv] = x[2*npq + 1:2*npq + npv]
    vang[polar.network.ref] = p[1:nref]
    return vang
end
function get(polar::PolarForm{T, VT, AT}, ::PS.Buses, ::PS.ActivePower, x, u, p; V=eltype(x)) where {T, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    vmag = get(polar, PS.Buses(), PS.VoltageMagnitude(), x, u, p)
    vang = get(polar, PS.Buses(), PS.VoltageAngle(), x, u, p)
    pinj = VT(undef, nbus)
    pinj[polar.network.pv] = u[nref + 1:nref + npv] - polar.active_load[polar.network.pv]
    pinj[polar.network.pq] = p[nref + 1:nref + npq]
    for bus in polar.network.ref
        pinj[bus] = PS.get_power_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
    end
    return pinj
end
function get(polar::PolarForm{T, VT, AT}, ::PS.Buses, ::PS.ReactivePower, x, u, p; V=eltype(x)) where {T, VT, AT}
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
function get(polar::PolarForm{T, VT, AT}, ::PS.Generator, ::PS.ActivePower, x, u, p; V=eltype(x)) where {T, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    pinj = VT(undef, nbus)
    pinj[polar.network.pv] = u[nref + 1:nref + npv]
    pinj[polar.network.pq] = p[nref + 1:nref + npq]
    for bus in polar.network.ref
        pinj[bus] = PS.get_power_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
    end
    return pinj
end

function bounds(polar::PolarForm, ::State)
    return polar.x_min, polar.x_max
end
function bounds(polar::PolarForm, ::Control)
    return polar.u_min, polar.u_max
end

function cost_production(polar::PolarForm, x, u, p)
    # indexes
    # for now, let's just return the sum of all generator power
    power_generations = get(polar, PS.Generator(), PS.ActivePower(), x, u, p)
    # TODO

    c0 = polar.costs_coefficients[:, 2]
    c1 = polar.costs_coefficients[:, 3]
    c2 = polar.costs_coefficients[:, 4]
    # Return quadratic cost
    cost = c0 .+ c1 .* power_generations + c2 .* power_generations.^2
    return cost
end

function get_network_state(polar::PolarForm{T, VT, AT}, x, u, p; V=Float64) where {T, VT, AT}
    nbus = PS.get(polar.network, PS.NumberOfBuses())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())
    nref = PS.get(polar.network, PS.NumberOfSlackBuses())
    pf = polar.network

    vmag = VT(undef, nbus)
    vang = VT(undef, nbus)
    pinj = VT(undef, nbus)
    qinj = VT(undef, nbus)

    vmag[pf.pq] .= x[1:npq]
    vang[pf.pq] .= x[npq + 1:2*npq]
    vang[pf.pv] .= x[2*npq + 1:2*npq + npv]

    vmag[pf.ref] .= u[1:nref]
    pinj[pf.pv] .= u[nref + 1:nref + npv] - polar.active_load[pf.pv]
    vmag[pf.pv] .= u[nref + npv + 1:nref + 2*npv]

    vang[pf.ref] .= p[1:nref]
    pinj[pf.pq] .= p[nref + 1:nref + npq]
    qinj[pf.pq] .= p[nref + npq + 1:nref + 2*npq]

    for bus in pf.ref
        pinj[bus] = PS.get_power_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
        qinj[bus] = PS.get_react_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
    end

    for bus in pf.pv
        qinj[bus] = PS.get_react_injection(bus, vmag, vang, polar.ybus_re, polar.ybus_im)
    end

    return vmag, vang, pinj, qinj
end

function powerflow(
    polar::PolarForm{T, VT, AT},
    x::VT,
    u::VT,
    p::VT;
    npartitions=2,
    solver="default",
    tol=1e-7,
    maxiter=20,
    verbose_level=0,
) where {T, VT, AT}
    # Retrieve parameter and initial voltage guess
    @timeit TIMER "Init" begin
        Vm, Va, pbus, qbus = get_network_state(polar, x, u, p)
    end
    V = convert(polar.AT{Complex{T}, 1}, polar.network.vbus)

    nbus = PS.get(polar.network, PS.NumberOfBuses())
    ngen = PS.get(polar.network, PS.NumberOfGenerators())
    n_states = get(polar, NumberOfState())
    npv = PS.get(polar.network, PS.NumberOfPVBuses())
    npq = PS.get(polar.network, PS.NumberOfPQBuses())

    # TODO: avoid moving pv and pq each time we are calling powerflow
    ref = convert(polar.AT{Int, 1}, polar.network.ref)
    pv = convert(polar.AT{Int, 1}, polar.network.pv)
    pq = convert(polar.AT{Int, 1}, polar.network.pq)

    # iteration variables
    iter = 0
    converged = false

    # indices
    j1 = 1
    j2 = npq
    j3 = j2 + 1
    j4 = j2 + npq
    j5 = j4 + 1
    j6 = j4 + npv

    # form residual function directly on target device
    F = VT(undef, n_states)
    dx = similar(F)
    fill!(F, zero(T))
    fill!(dx, zero(T))

    # Evaluate residual function
    ExaPF.residualFunction_polar!(F, Vm, Va,
                            polar.ybus_re, polar.ybus_im,
                            pbus, qbus, pv, pq, nbus)
    # Build the AD Jacobian structure
    stateJacobianAD = AD.StateJacobianAD(F, Vm, Va,
                                         polar.ybus_re, polar.ybus_im, pbus, qbus, pv, pq, ref, nbus)
    designJacobianAD = AD.DesignJacobianAD(F, Vm, Va,
                                           polar.ybus_re, polar.ybus_im, pbus, qbus, pv, pq, ref, nbus)
    if verbose_level >= ExaPF.VERBOSE_LEVEL_MEDIUM
        print("State Jacobian  --- ")
        println(stateJacobianAD)
        print("Design Jacobian --- ")
        println(designJacobianAD)
    end

    J = stateJacobianAD.J
    preconditioner = Precondition.NoPreconditioner()
    if solver != "default"
        nblock = size(J,1) / npartitions
        if verbose_level >= ExaPF.VERBOSE_LEVEL_MEDIUM
            println("Blocks: $npartitions, Blocksize: n = ", nblock,
                    " Mbytes = ", (nblock*nblock*npartitions*8.0)/1024.0/1024.0)
            println("Partitioning...")
        end
        preconditioner = Precondition.Preconditioner(J, npartitions, device)
        if verbose_level >= ExaPF.VERBOSE_LEVEL_MEDIUM
            println("$npartitions partitions created")
        end
    end

    # check for convergence
    normF = norm(F, Inf)
    if verbose_level >= ExaPF.VERBOSE_LEVEL_HIGH
        @printf("Iteration %d. Residual norm: %g.\n", iter, normF)
    end
    if normF < tol
        converged = true
    end

    linsol_iters = Int[]
    Vapv = view(Va, pv)
    Vapq = view(Va, pq)
    Vmpq = view(Vm, pq)
    dx12 = view(dx, j1:j2)
    dx34 = view(dx, j3:j4)
    dx56 = view(dx, j5:j6)

    @timeit TIMER "Newton" while ((!converged) && (iter < maxiter))

        iter += 1

        @timeit TIMER "Jacobian" begin
            AD.residualJacobianAD!(stateJacobianAD, ExaPF.residualFunction_polar!, Vm, Va,
                                   polar.ybus_re, polar.ybus_im, pbus, qbus, pv, pq, ref, nbus, TIMER)
        end
        J = stateJacobianAD.J

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

        @timeit TIMER "Exponential" V .= Vm .* exp.(1im .* Va)

        @timeit TIMER "Angle and magnitude" begin
            ExaPF.polar!(Vm, Va, V, polar.device)
        end

        F .= 0.0
        @timeit TIMER "Residual function" begin
            ExaPF.residualFunction_polar!(F, Vm, Va,
                polar.ybus_re, polar.ybus_im,
                pbus, qbus, pv, pq, nbus)
        end

        @timeit TIMER "Norm" normF = norm(F, Inf)
        if verbose_level >= ExaPF.VERBOSE_LEVEL_HIGH
            @printf("Iteration %d. Residual norm: %g.\n", iter, normF)
        end

        if normF < tol
            converged = true
        end
    end

    if verbose_level >= ExaPF.VERBOSE_LEVEL_HIGH
        if converged
            @printf("N-R converged in %d iterations.\n", iter)
        else
            @printf("N-R did not converge.\n")
        end
    end

    xk = get(polar, State(), Vm, Va, pbus, qbus)

    # Timer outputs display
    if verbose_level >= ExaPF.VERBOSE_LEVEL_MEDIUM
        show(TIMER)
        println("")
    end
    reset_timer!(TIMER)
    conv = ExaPF.ConvergenceStatus(converged, iter, normF, sum(linsol_iters))
    return xk, conv
end

