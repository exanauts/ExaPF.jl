using CUDA
using ExaPF
using KernelAbstractions
using Test
import ExaPF: PowerSystem
using UnicodePlots
#datafile = joinpath(dirname(@__FILE__), "..", "data", "case9241pegase.m")
#datafile = joinpath(dirname(@__FILE__), "..", "data", "case118.m")
#datafile = joinpath(dirname(@__FILE__), "..", "data", "case300.m")
#datafile = joinpath(dirname(@__FILE__), "..", "data", "case14.m")
#datafile = joinpath(dirname(@__FILE__), "..", "data", "case9.m")
datafile = joinpath(dirname(@__FILE__), "..", "data", "case_ACTIVSg70k.m")
#datafile = joinpath(dirname(@__FILE__), "..", "data", "case1354pegase.m")
datafile = joinpath(dirname(@__FILE__), "..", "data", "caseGO13R-15.raw")
#datafile = joinpath(dirname(@__FILE__), "..", "data", "caseGO30R-025.raw")

pf = PowerSystem.PowerNetwork(datafile)
npartitions = 200
tolerance = 1e-6
#device = CUDADevice()
device = CPU()
polar = PolarForm(pf, device)
jac = ExaPF.residual_jacobian(State(), polar)
println(spy(jac, height=20, width=40))

if true
#precond = ExaPF.LinearSolvers.BlockJacobiPreconditioner(jac, npartitions, device)
#println(spy(precond.P, height=20, width=40))
x0 = ExaPF.initial(polar, State())
uk = ExaPF.initial(polar, Control())
#algo = ExaPF.LinearSolvers.EigenBICGSTAB(precond)
#algo = ExaPF.LinearSolvers.KrylovBICGSTAB(precond; verbose=0, rtol=1e-10, atol=1e-10)
algo = ExaPF.LinearSolvers.DirectSolver()
xk = copy(x0)
nlp = ExaPF.ReducedSpaceEvaluator(
    polar, xk, uk;
    powerflow_solver=NewtonRaphson(tol=tolerance, verbose=3),
    linear_solver=algo
)
convergence = ExaPF.update!(nlp, uk)
end
