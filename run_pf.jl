using ExaPF
using CUDA
using CUDSS
using LinearAlgebra
const PS = ExaPF.PowerSystem

polar = ExaPF.load_polar("case9.m")
stack = ExaPF.NetworkStack(polar)
nbus = PS.get(polar, PS.NumberOfBuses())
nscen = 10
ploads = rand(nbus, nscen)
qloads = rand(nbus, nscen)
blk_polar = ExaPF.BlockPolarForm(polar, nscen)
blk_stack = ExaPF.NetworkStack(blk_polar)
ExaPF.set_params!(blk_stack, ploads, qloads)
powerflow = ExaPF.PowerFlowBalance(blk_polar) ∘ ExaPF.Basis(blk_polar);
blk_jx = ExaPF.BatchJacobian(blk_polar, powerflow, State());
ExaPF.set_params!(blk_jx, blk_stack);
ExaPF.jacobian!(blk_jx, blk_stack);
conv = ExaPF.nlsolve!(
    NewtonRaphson(verbose=2),
    blk_jx,
    blk_stack;
)

sol_ref = blk_stack.input[blk_jx.map]
res = run_pf("case9.m", CPU(), :block_polar, 10, ploads, qloads; verbose=2)
sol_cpu = get_sol(res)
isapprox(norm(sol_cpu), norm(sol_ref))
res = run_pf("case9.m", CUDABackend(), :block_polar, 10, ploads, qloads; verbose=2, batch_linear_solver=true)
sol_gpu = get_sol(res)

isapprox(norm(sol_gpu), norm(sol_ref))
