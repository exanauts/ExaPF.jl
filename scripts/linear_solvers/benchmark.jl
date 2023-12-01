
using CUDA
using KernelAbstractions

using ExaPF
import ExaPF: AutoDiff

using LazyArtifacts
using LinearAlgebra
using KrylovPreconditioners
using KLU
using CUSOLVERRF

const LS = ExaPF.LinearSolvers

# KLU wrapper
LinearAlgebra.lu!(K::KLU.KLUFactorization, J) = KLU.klu!(K, J)

function build_instance(datafile, device)
    polar = ExaPF.PolarForm(datafile, device)
    stack = ExaPF.NetworkStack(polar)
    # Instantiate Automatic Differentiation
    pflow = ExaPF.PowerFlowBalance(polar) ∘ ExaPF.PolarBasis(polar)
    jx = ExaPF.Jacobian(polar, pflow, State())
    return (
        model=polar,
        jacobian=jx,
        stack=stack,
    )
end

function benchmark_cpu_klu(datafile, pf_solver; ntrials=3)
    instance = build_instance(datafile, CPU())
    # Initiate KLU
    klu_factorization = KLU.klu(instance.jacobian.J)
    klu_solver = LS.DirectSolver(klu_factorization)
    # Solve power flow
    tic = 0.0
    for _ in 1:ntrials
        ExaPF.init!(instance.model, instance.stack) # reinit stack
        tic += @elapsed ExaPF.nlsolve!(pf_solver, instance.jacobian, instance.stack; linear_solver=klu_solver)
    end
    return tic / ntrials
end

function benchmark_gpu_cusolverrf(datafile, pf_solver; ntrials=3)
    instance = build_instance(datafile, CUDABackend())
    # Initiate CUSOLVERRF
    rf_factorization = CUSOLVERRF.RFLU(instance.jacobian.J)
    rf_solver = LS.DirectSolver(rf_factorization)
    # Solve power flow
    tic = 0.0
    for _ in 1:ntrials
        ExaPF.init!(instance.model, instance.stack) # reinit stack
        tic += @elapsed ExaPF.nlsolve!(pf_solver, instance.jacobian, instance.stack; linear_solver=rf_solver)
    end
    return tic / ntrials
end

function benchmark_gpu_krylov(datafile, pf_solver; ntrials=3)
    instance = build_instance(datafile, CUDABackend())
    # Build Krylov solver
    n_blocks = 32
    n_states = size(instance.jacobian, 1)
    n_partitions = div(n_states, n_blocks)
    jac_gpu = instance.jacobian.J
    precond = BlockJacobiPreconditioner(jac_gpu, n_partitions, CUDABackend(), 0)
    krylov_solver = ExaPF.KrylovBICGSTAB(
        jac_gpu; P=precond, ldiv=false, scaling=true,
        rtol=1e-7, atol=1e-7, verbose=0,
    )
    # Solve power flow
    tic = 0.0
    for _ in 1:ntrials
        ExaPF.init!(instance.model, instance.stack) # reinit stack
        tic += @elapsed ExaPF.nlsolve!(pf_solver, instance.jacobian, instance.stack; linear_solver=krylov_solver)
    end
    return tic / ntrials
end

pf_algo = NewtonRaphson(; verbose=0, tol=1e-7)
datafile = joinpath(artifact"ExaData", "ExaData", "case9241pegase.m")

time_klu = benchmark_cpu_klu(datafile, pf_algo)
time_cusolverf = benchmark_gpu_cusolverrf(datafile, pf_algo)
time_krylov = benchmark_gpu_krylov(datafile, pf_algo)

println("Benchmark powerflow with $(basename(datafile)):")
println("    > KLU (s)           : ", time_klu)
println("    > CUSOLVERRF (s)    : ", time_cusolverf)
println("    > KRYLOV (s)        : ", time_krylov)
