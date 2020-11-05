using CUDA
using ExaPF
using KernelAbstractions
using Test
using Printf

import ExaPF: PowerSystem, LinearSolvers, TimerOutputs

# We do need the time in ms
function ExaPF.TimerOutputs.prettytime(t)
    value, units = t / 1e6, "ms"

    if round(value) >= 100
        str = string(@sprintf("%.0f", value), units)
    elseif round(value * 10) >= 100
        str = string(@sprintf("%.1f", value), units)
    elseif round(value * 100) >= 100
        str = string(@sprintf("%.2f", value), units)
    else
        str = string(@sprintf("%.3f", value), units)
    end
    return lpad(str, 6, " ")
end


@show case = ARGS[1]
setup = parse(Int, ARGS[2])
if setup == 1
    device = CPU()
    linsolver = LinearSolvers.DirectSolver
elseif setup == 2
    device = CUDADevice()
    linsolver = LinearSolvers.DirectSolver
elseif setup == 3
    device = CUDADevice()
    linsolver = LinearSolvers.BICGSTAB
else
    error("No valid setup selected")
end

datafile = joinpath(dirname(@__FILE__), case)
if endswith(datafile, ".m")
    pf = PowerSystem.PowerNetwork(datafile, 1)
else
    pf = PowerSystem.PowerNetwork(datafile)
end
# Parameters
tolerance = 1e-6
polar = PolarForm(pf, device)
jac = ExaPF._state_jacobian(polar)
@show size(jac)
@show npartitions = ceil(Int64,(size(jac,1)/64))
precond = ExaPF.LinearSolvers.BlockJacobiPreconditioner(jac, npartitions, device)
# Retrieve initial state of network
x0 = ExaPF.initial(polar, State())
uk = ExaPF.initial(polar, Control())
p = ExaPF.initial(polar, Parameters())

algo = linsolver(precond)
xk = copy(x0)
nlp = ExaPF.ReducedSpaceEvaluator(polar, xk, uk, p;
                                    ε_tol=tolerance, linear_solver=algo)
convergence = ExaPF.update!(nlp, uk; verbose_level=ExaPF.VERBOSE_LEVEL_HIGH)
nlp.x .= x0                                   
convergence = ExaPF.update!(nlp, uk; verbose_level=ExaPF.VERBOSE_LEVEL_HIGH)
nlp.x .= x0                                   
convergence = ExaPF.update!(nlp, uk; verbose_level=ExaPF.VERBOSE_LEVEL_HIGH)
@show convergence
