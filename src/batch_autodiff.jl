module BatchAutoDiff

using SparseArrays

using CUDA
import CUDA.CUSPARSE
import ForwardDiff
import SparseDiffTools
using KernelAbstractions

using ..ExaPF: Spmat, xzeros, State, Control
using ..ExaPF: AutoDiff

@kernel function seed_kernel!(
    duals::AbstractArray{ForwardDiff.Dual{T, V, N}}, x,
    seeds::AbstractArray{ForwardDiff.Partials{N, V}}
) where {T,V,N}
    i = @index(Group, Linear)
    j = @index(Local, Linear)
    duals[j, i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[j, i])
end

"""
    seed!

Calling the seeding kernel.
Seeding is parallelized over the `ncolor` number of duals.

"""
function seed!(t1sseeds, varx, t1svarx)
    if isa(t1sseeds, Matrix)
        device = CPU()
        kernel! = seed_kernel!(CPU())
    else
        device = CUDADevice()
        kernel! = seed_kernel!(CUDADevice())
    end
    nbatch = size(t1sseeds, 1)
    nvars = size(t1sseeds, 2)
    ndrange = (nbatch, nvars)
    ev = kernel!(t1svarx, varx, t1sseeds, ndrange=ndrange, dependencies=Event(device))
    wait(ev)
end

# Get partials for Hessian projection
@kernel function getpartials_hv_kernel!(hv, adj_t1sx, map)
    i = @index(Group, Linear)
    j = @index(Local, Linear)
    hv[j, i] = ForwardDiff.partials(adj_t1sx[j, map[i]]).values[1]
end

"""
    getpartials_kernel!(compressedJ, t1sF)

Calling the partial extraction kernel.
Extract the partials from the AutoDiff dual type on the target
device and put it in the compressed Jacobian `compressedJ`.

"""
function getpartials_kernel!(hv::AbstractMatrix, adj_t1sx, map)
    if isa(hv, Matrix)
        device = CPU()
        kernel! = getpartials_hv_kernel!(CPU())
    else
        device = CUDADevice()
        kernel! = getpartials_hv_kernel!(CUDADevice())
    end
    nbatch = size(hv, 1)
    nvars = size(hv, 2)
    ndrange = (nbatch, nvars)
    ev = kernel!(hv, adj_t1sx, map, ndrange=ndrange, dependencies=Event(device))
    wait(ev)
end

end
