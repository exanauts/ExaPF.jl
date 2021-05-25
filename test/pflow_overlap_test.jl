using CUDA
using ExaPF
using KernelAbstractions
using Test

import ExaPF: PowerSystem

function foo(overlap=0)
    #datafile = joinpath(dirname(@__FILE__), "data", "case9.m")
    #datafile = joinpath(dirname(@__FILE__), "data", "case14.m")
    datafile = joinpath(dirname(@__FILE__), "data", "case300.m")
    #datafile = joinpath(dirname(@__FILE__), "data", "case9241pegase.m")
    
    pf = PowerSystem.PowerNetwork(datafile)
    npartitions = 200
    tolerance = 1e-6

    dev = CPU()
    dev = CUDADevice()
    polar = PolarForm(pf, dev)
    jac = ExaPF.jacobian_sparsity(polar, ExaPF.power_balance, State())
    precond = ExaPF.LinearSolvers.BlockJacobiPreconditioner(jac, npartitions, dev, overlap)
    x0 = ExaPF.initial(polar, State())
    uk = ExaPF.initial(polar, Control())
    algo = ExaPF.LinearSolvers.KrylovBICGSTAB(precond; verbose=0, rtol=1e-10, atol=1e-10)
    xk = copy(x0)
    nlp = ExaPF.ReducedSpaceEvaluator(
        polar, xk, uk;
        powerflow_solver=NewtonRaphson(tol=tolerance, verbose=2),
        linear_solver=algo
    )
    convergence = ExaPF.update!(nlp, uk)
end
foo()

