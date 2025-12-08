
using Random
using CUDA
using KernelAbstractions

using ExaPF
import ExaPF: AutoDiff

using LazyArtifacts
using LinearAlgebra
using KrylovPreconditioners
using KLU
using CUDSS

const LS = ExaPF.LinearSolvers
const PS = ExaPF.PowerSystem

# KLU wrapper
LinearAlgebra.lu!(K::KLU.KLUFactorization, J) = KLU.klu!(K, J)

function generate_loads(model, stack, n_blocks, magnitude)
    nbus = get(model, ExaPF.PS.NumberOfBuses())
    pload_det = get(model.network, PS.ActiveLoad())
    qload_det = get(model.network, PS.ReactiveLoad())

    has_load = (pload_det .> 0)

    Random.seed!(1)
    pload = magnitude .* (randn(nbus, n_blocks) .* has_load) .+ pload_det
    qload = magnitude .* (randn(nbus, n_blocks) .* has_load) .+ qload_det
    return pload, qload
end

function build_instance(datafile, n_blocks, backend, magnitude)
    polar = ExaPF.BlockPolarForm(datafile, n_blocks, backend)
    stack = ExaPF.NetworkStack(polar)
    pload, qload = generate_loads(polar, stack, n_blocks, magnitude)
    # Load scenarios into stacks
    ExaPF.set_params!(stack, pload, qload)
    # Instantiate Automatic Differentiation
    pflow = ExaPF.PowerFlowBalance(polar) ∘ ExaPF.PolarBasis(polar)
    jx = ExaPF.BatchJacobian(polar, pflow, State())
    ExaPF.set_params!(jx, stack)
    ExaPF.jacobian!(jx, stack)
    return (
        model=polar,
        jacobian=jx,
        stack=stack,
    )
end

function benchmark_cpu_klu(datafile, n_blocks, pf_solver; ntrials=3, magnitude=0.01)
    instance = build_instance(datafile, n_blocks, CPU(), magnitude)
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

function benchmark_gpu_cudss(datafile, n_blocks, pf_solver; ntrials=3, magnitude=0.01)
    instance = build_instance(datafile, n_blocks, CUDABackend(), magnitude)
    # Initiate CUDSS
    J = instance.jacobian.J
    cudss_factorization = CUDSS.lu(J)
    cudss_solver = LS.DirectSolver(cudss_factorization)
    # Solve power flow
    tic = 0.0
    for _ in 1:ntrials
        ExaPF.init!(instance.model, instance.stack) # reinit stack
        tic += @elapsed ExaPF.nlsolve!(pf_solver, instance.jacobian, instance.stack; linear_solver=cudss_solver)
    end
    return tic / ntrials
end

function benchmark_gpu_krylov(datafile, n_scenarios, pf_solver; ntrials=3, magnitude=0.01)
    instance = build_instance(datafile, n_scenarios, CUDABackend(), magnitude)
    # Build Krylov solver
    n_blocks = 32 * n_scenarios
    n_states = size(instance.jacobian.J, 1)
    n_partitions = div(n_states, n_blocks)
    jac_gpu = instance.jacobian.J
    precond = BlockJacobiPreconditioner(jac_gpu, n_partitions, CUDABackend(), 0)
    krylov_solver = ExaPF.Bicgstab(
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

n_scenarios = 10
pf_algo = NewtonRaphson(; verbose=0, tol=1e-7)
datafile = joinpath(artifact"ExaData", "ExaData", "case1354pegase.m")

time_klu = benchmark_cpu_klu(datafile, n_scenarios, pf_algo)
time_cudss = benchmark_gpu_cudss(datafile, n_scenarios, pf_algo)
time_krylov = benchmark_gpu_krylov(datafile, n_scenarios, pf_algo)

println("Benchmark powerflow with $(basename(datafile)):")
println("    > KLU (s)           : ", time_klu)
println("    > CUDSS (s)         : ", time_cudss)
println("    > KRYLOV (s)        : ", time_krylov)
