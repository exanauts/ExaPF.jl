using CUDA
using KernelAbstractions
using ExaPF
import ExaPF: AutoDiff
using LazyArtifacts
using LinearAlgebra
using KrylovPreconditioners
using PProf
using Profile


const PS = ExaPF.PowerSystem
const LS = ExaPF.LinearSolvers
# datafile = joinpath(artifact"ExaData", "ExaData", "matpower", "case_ACTIVSg70k.m")
# datafile = joinpath(artifact"ExaData", "ExaData", "case1354.m")
datafile = joinpath(artifact"ExaData", "ExaData", "case9241pegase.m")
polar = ExaPF.PolarForm(datafile, CPU())
stack = ExaPF.NetworkStack(polar)
ExaPF.init!(polar, stack)
@time convergence = run_pf(polar, stack; verbose=1)

pf = PS.PowerNetwork(datafile)
polar_gpu = ExaPF.PolarForm(pf, CUDABackend())
stack_gpu = ExaPF.NetworkStack(polar_gpu)
basis_gpu = ExaPF.PolarBasis(polar_gpu)
pflow_gpu = ExaPF.PowerFlowBalance(polar_gpu) ∘ basis_gpu
mapx = ExaPF.mapping(polar, State());
jx_gpu = ExaPF.Jacobian(polar_gpu, pflow_gpu, mapx)
direct_linear_solver = LS.DirectSolver(jx_gpu.J)
pf_algo = NewtonRaphson(; verbose=1, tol=1e-10)


jac_gpu = jx_gpu.J;

npartitions = div(size(jac_gpu,1), 32);
precond = BlockJacobiPreconditioner(jac_gpu, npartitions, CUDABackend(), 0);
# precond = KrylovPreconditioners.kp_ilu0(jac_gpu)
# linear_solver = ExaPF.KrylovBICGSTAB(jac_gpu; P=precond, ldiv=false, scaling=true, atol=1e-10, verbose=0);
# linear_solver = ExaPF.KrylovBICGSTAB(jac_gpu; P=precond, ldiv=false, scaling=false, atol=1e-10, verbose=0);
# linear_solver = ExaPF.KrylovBICGSTAB(jac_gpu; P=precond, ldiv=false, scaling=false, atol=1e-10, verbose=0, maxiter=500);
linear_solver = ExaPF.KrylovBICGSTAB(
    jac_gpu; P=precond, ldiv=false, scaling=true,
    rtol=1e-7, atol=1e-7, verbose=0
);
pf_algo = NewtonRaphson(; verbose=1, tol=1e-7, maxiter=20)
ExaPF.init!(polar_gpu, stack_gpu)
reset_timer!(linear_solver.precond)
@time convergence = ExaPF.nlsolve!(pf_algo, jx_gpu, stack_gpu; linear_solver=linear_solver)
@show get_timer(linear_solver.precond)
ExaPF.init!(polar_gpu, stack_gpu)
@time convergence = ExaPF.nlsolve!(pf_algo, jx_gpu, stack_gpu; linear_solver=direct_linear_solver)
@show linear_solver.inner.stats.niter
@show convergence
# Profiling
ExaPF.init!(polar_gpu, stack_gpu)
Profile.clear()
Profile.@profile begin
    linear_solver.precond.timer_update = 0.0
    convergence = ExaPF.nlsolve!(pf_algo, jx_gpu, stack_gpu; linear_solver=linear_solver)
    @show linear_solver.precond.timer_update
end
Profile.clear()
ExaPF.init!(polar_gpu, stack_gpu)
Profile.@profile begin
    linear_solver.precond.timer_update = 0.0
    convergence = ExaPF.nlsolve!(pf_algo, jx_gpu, stack_gpu; linear_solver=linear_solver)
    @show linear_solver.precond.timer_update
end
PProf.pprof()
@show convergence
