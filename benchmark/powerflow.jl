using ExaPF
using Printf
using SparseArrays
using TimerOutputs
import ExaPF: LinearSolvers
const LS = LinearSolvers

using CUDA
using KernelAbstractions
if has_cuda_gpu()
    # Load CUDA related code for sparse direct linear algebra
    include(joinpath(dirname(@__FILE__), "..", "test", "cusolver.jl"))
end

# By default, use instances stored in ExaPF
INSTANCES_DIR = joinpath(dirname(@__FILE__), "..", "data")

INSTANCES = [
    "case9.m",
    "case30.m",
    "case57.m",
    "case300.m",
    "case1354.m",
    "case9241pegase.m",
]

function benchmark_single_instance(name, device)
    pf_solver = NewtonRaphson(tol=1e-8, maxiter=10, verbose=2)

    # Instantiate
    polar = ExaPF.PolarForm(name, device)
    buffer = ExaPF.get(polar, ExaPF.PhysicalState())

    # AutoDiff
    jx = ExaPF.AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
    # Linear solver
    SpMT = isa(device, CPU) ? SparseMatrixCSC : CUSPARSE.CuSparseMatrixCSR
    J = ExaPF.powerflow_jacobian(polar) |> SpMT
    linear_solver = LS.DirectSolver(J)

    ExaPF.init_buffer!(polar, buffer)
    reset_timer!(ExaPF.TIMER)
    @time ExaPF.powerflow(polar, jx, buffer, pf_solver; linear_solver=linear_solver)
    return
end

function benchmark_powerflow_solver(instances, device; ntrials=10)
    # Match option in MATPOWER
    pf_solver = NewtonRaphson(tol=1e-8, maxiter=10)
    benchtimes = zeros(length(instances))

    for (j, name) in enumerate(instances)
        println("Benchmark power flow on $name")
        # Instantiate
        instance = joinpath(INSTANCES_DIR, name)
        polar = ExaPF.PolarForm(instance, device)
        buffer = ExaPF.get(polar, ExaPF.PhysicalState())

        # AutoDiff
        jx = ExaPF.AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
        # Linear solver
        SpMT = isa(device, CPU) ? SparseMatrixCSC : CUSPARSE.CuSparseMatrixCSR
        J = ExaPF.powerflow_jacobian(polar) |> SpMT
        linear_solver = LS.DirectSolver(J)

        pf_time = 0.0
        for i in 1:(ntrials+1)
            ExaPF.init_buffer!(polar, buffer)
            t = @timed ExaPF.powerflow(polar, jx, buffer, pf_solver; linear_solver=linear_solver)
            if i > 1 # discard first compilation
                pf_time += t.time
            end
        end
        benchtimes[j] = pf_time / ntrials
    end

    return benchtimes
end

@info "Benchmark CPU"
res_cpu = benchmark_powerflow_solver(INSTANCES, CPU())
if has_cuda_gpu()
    @info "Benchmark CUDA GPU"
    res_gpu = benchmark_powerflow_solver(INSTANCES, CUDADevice())
else
    res_gpu = fill(NaN, length(INSTANCES))
end

# Display results
@printf("%s\n", "-"^42)
@printf("%-20s | %8s | %8s\n", "case", "CPU", "CUDA")
@printf("%s\n", "-"^42)
for (i, name) in enumerate(INSTANCES)
    @printf("%-20s | %8.5f | %8.5f\n", name, res_cpu[i], res_gpu[i])
end
@printf("%s\n", "-"^42)
