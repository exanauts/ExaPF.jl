module ExaPFApp

using ExaPF
using KernelAbstractions
# TODO: uncomment this when https://github.com/JuliaGPU/GPUCompiler.jl/issues/611 is fixed
using CUDA, CUDSS
using SparseArrays: sprand

using PrecompileTools

# Note: CUDA precompilation workloads are disabled for juliac native compilation
# as NVVM intrinsics cannot be compiled to native CPU code.
# @compile_workload begin
#     # Force compilation of CUDA sparse matrix methods
#     if CUDA.functional()
#         dummy = CUDA.CUSPARSE.CuSparseMatrixCSR(sprand(Float64, 4, 4, 0.5))
#     end
# end

function run_pf_gpu(case::String)
    println("Using GPU Backend (CUDA + CUDSS)")
    run_pf(case, CUDABackend(), :polar; verbose=2)
end

function run_pf_cpu(case::String)
    run_pf(case, CPU(), :polar; verbose=2)
end

function @main(args::Vector{String})::Int
    if args[2] == "cpu"
        run_pf_cpu(args[1])
    elseif args[2] == "gpu"
        run_pf_gpu(args[1])
    else
        @error "Unknown backend: $(args[2]). Use 'cpu' or 'gpu'."
        return 1
    end
    return 0
end

end # module
