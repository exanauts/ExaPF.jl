module ExaPFApp

using ExaPF
using KernelAbstractions
# TODO: uncomment this when https://github.com/JuliaGPU/GPUCompiler.jl/issues/611 is fixed
# using CUDA, CUDSS

# function run_pf_gpu(case::String)
#     println("Using GPU Backend (CUDA + CUDSS)")
#     run_pf(case, CUDABackend(), :polar; verbose=2)
# end

function run_pf_cpu(case::String)
    run_pf(case, CPU(), :polar; verbose=2)
end

function @main(args::Vector{String})::Int
    if args[2] == "cpu"
        run_pf_cpu(args[1])
    elseif args[2] == "gpu"
        @error "GPU backend is not supported yet"
        return 1
        # run_pf_gpu(args[1])
    else
        @error "Unknown backend: $(args[2]). Use 'cpu' or 'gpu'."
        return 1
    end
    return 0
end

end # module
