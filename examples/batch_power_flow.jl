using ExaPF
using CUDA
using CUDSS
using LinearAlgebra
const PS = ExaPF.PowerSystem

res = run_pf("case9.m", CPU(), :block_polar, 10, ploads, qloads; verbose=2)
sol_cpu = get_solution(res)
isapprox(norm(sol_cpu), norm(sol_ref))
res = run_pf("case9.m", CUDABackend(), :block_polar, 10, ploads, qloads; verbose=2)
sol_gpu = get_solution(res)

isapprox(norm(sol_gpu), norm(sol_ref))
