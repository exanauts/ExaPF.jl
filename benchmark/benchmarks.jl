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
    mapx = ExaPF.my_map(polar, State())
    nx = length(mapx)
    stack = ExaPF.NetworkStack(polar)

    basis = ExaPF.PolarBasis(polar)
    pflow = ExaPF.PowerFlowBalance(polar)
    jx = ExaPF.MyJacobian(polar, pflow ∘ basis, mapx)
    J = jx.J
    npartitions = ceil(Int64,(size(jx.J,1)/64))
    if npartitions < 2
        npartitions = 2
    end
    precond = ExaPF.LinearSolvers.BlockJacobiPreconditioner(J, npartitions, device)

    algo = linsolver(J; P=precond)
    powerflow_solver = NewtonRaphson(tol=ntol)
    VT = typeof(stack.input)
    pf_buffer = ExaPF.NLBuffer{VT}(nx)

    # Warm-up
    ExaPF.nlsolve!(
        powerflow_solver, jx, stack; linear_solver=algo, nl_buffer=pf_buffer,
    )

    ExaPF.init!(polar, stack)
    res = @timed ExaPF.nlsolve!(
        powerflow_solver, jx, stack; linear_solver=algo, nl_buffer=pf_buffer,
    )
    convergence = res.value

    return convergence.has_converged, res.time
end

function main()
    linsolver = eval(Meta.parse("LinearSolvers.$(ARGS[1])"))
    device = eval(Meta.parse("$(ARGS[2])()"))
    datafile = joinpath(dirname(@__FILE__), ARGS[3])

    has_converged, timer = run_benchmark(datafile, device, linsolver)
    @test has_converged

    println("$(ARGS[1]), $(ARGS[2]), $(ARGS[3]),",
            timer,
            ", $(has_converged)")

end

main()

