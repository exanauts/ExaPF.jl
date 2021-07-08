using CUDA
using ExaPF
using KernelAbstractions
using Test
using Printf
using TimerOutputs

import ExaPF: PowerSystem, LinearSolvers, AutoDiff

# Newton-Raphson tolerance
# For debugging in REPL use the following lines
# empty!(ARGS)
# push!(ARGS, "KrylovBICGSTAB")
# push!(ARGS, "CPU")
# push!(ARGS, "case300.m")
# push!(ARGS, "caseGO30R-025.raw")


# We do need the time in ms, and not with time units all over the place
function TimerOutputs.prettytime(t)
    value = t / 1e6 # "ms"

    if round(value) >= 100
        str = string(@sprintf("%.0f", value))
    elseif round(value * 10) >= 100
        str = string(@sprintf("%.1f", value))
    elseif round(value * 100) >= 100
        str = string(@sprintf("%.2f", value))
    else
        str = string(@sprintf("%.3f", value))
    end
    return lpad(str, 6, " ")
end

function printtimer(timers, key::String)
   TimerOutputs.prettytime(timers[key].accumulated_data.time)
end

function run_benchmark(datafile, device, linsolver)
    ntol = 1e-6
    pf = PowerSystem.PowerNetwork(datafile)
    polar = PolarForm(pf, device)
    cache = ExaPF.get(polar, ExaPF.PhysicalState())
    jx = AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
    J = jx.J
    npartitions = ceil(Int64,(size(jx.J,1)/64))
    if npartitions < 2
        npartitions = 2
    end
    precond = ExaPF.LinearSolvers.BlockJacobiPreconditioner(J, npartitions, device)
    # Retrieve initial state of network
    u0 = ExaPF.initial(polar, Control())

    algo = linsolver(J; P=precond)
    powerflow_solver = NewtonRaphson(tol=ntol)
    nlp = ExaPF.ReducedSpaceEvaluator(polar;
                                      powerflow_solver=powerflow_solver)
    nlp.linear_solver = algo
    convergence = ExaPF.update!(nlp, u0)
    ExaPF.reset!(nlp)
    convergence = ExaPF.update!(nlp, u0)
    ExaPF.reset!(nlp)
    TimerOutputs.reset_timer!(ExaPF.TIMER)
    convergence = ExaPF.update!(nlp, u0)

    # Make sure we are converged
    @assert(convergence.has_converged)

    # Output
    prettytime = TimerOutputs.prettytime
    timers = ExaPF.TIMER.inner_timers
    inner_timer = timers["Newton"]
    return convergence.has_converged, timers, inner_timer
end

function main()
    linsolver = eval(Meta.parse("LinearSolvers.$(ARGS[1])"))
    device = eval(Meta.parse("$(ARGS[2])()"))
    datafile = joinpath(dirname(@__FILE__), ARGS[3])

    has_converged, timers, inner_timer = run_benchmark(datafile, device, linsolver)

    if ARGS[1] == "DirectSolver"
        println("$(ARGS[1]), $(ARGS[2]), $(ARGS[3]),",
                printtimer(timers, "Newton"),
                ", $(has_converged)")
    else
        println("$(ARGS[1]), $(ARGS[2]), $(ARGS[3]),",
                printtimer(timers, "Newton"),",",
                printtimer(inner_timer, "Jacobian"),",",
                printtimer(inner_timer, "Linear Solver"),
                ", $(has_converged)")
    end
    @test has_converged
end

main()

