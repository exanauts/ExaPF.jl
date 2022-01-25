using ExaPF
using Test
using KernelAbstractions
const LS = ExaPF.LinearSolvers
using SparseArrays
using CUDA
using CUDA.CUSPARSE
using CUDAKernels

USEGPU = 1


if USEGPU == 0
    localdevice = CPU()
    SMT = SparseMatrixCSC
else
    localdevice = CUDADevice()
    SMT = CuSparseMatrixCSR
end

#casefile = "examples/case1354pegase.m"
#casefile = "examples/case14.m"
casefile = "examples/case9241pegase.m"

println("Casefile: ", casefile)

polar = ExaPF.PolarForm(casefile, localdevice)

stack = ExaPF.NetworkStack(polar)
mapx = ExaPF.my_map(polar, State())
pf_solver = NewtonRaphson(tol=1e-6, verbose=2)
npartitions = 700
#npartitions = 2

basis = ExaPF.PolarBasis(polar)
pflow = ExaPF.PowerFlowBalance(polar)
n = length(pflow)

# Get reduced space Jacobian on the CPU
J = ExaPF.jacobian_sparsity(polar, pflow)
J = J[:, mapx]


# Build preconditioner
precond = LS.BlockJacobiPreconditioner(J, npartitions, localdevice, 1)

J_gpu = J |> SMT

# Init AD
jx = ExaPF.Jacobian(polar, pflow ∘ basis, mapx)

LinSolver = ExaPF.LinearSolvers.KrylovBICGSTAB
algo = LinSolver(J_gpu; P=precond)
ExaPF.init!(polar, stack)
convergence = ExaPF.nlsolve!(pf_solver, jx, stack; linear_solver=algo)
