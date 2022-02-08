using LazyArtifacts
using SparseArrays

using KernelAbstractions
using CUDA
using CUDAKernels

using ExaPF

const LS = ExaPF.LinearSolvers

const INSTANCES_DIR = joinpath(artifact"ExaData", "ExaData")

USEGPU = 1


if USEGPU == 0
    localdevice = CPU()
else
    localdevice = CUDADevice()
end

case = "case1354pegase.m"
casefile = joinpath(INSTANCES_DIR, case)

println("Casefile: ", casefile)

#=
    Load data
=#
# Load instance
polar = ExaPF.PolarForm(casefile, localdevice)
# Load variables
stack = ExaPF.NetworkStack(polar)
# Mapping associated to the state
mapx = ExaPF.mapping(polar, State())
# Power flow solver
pf_solver = NewtonRaphson(tol=1e-6, verbose=2)
# Expressions
basis = ExaPF.PolarBasis(polar)
pflow = ExaPF.PowerFlowBalance(polar)
# Init AD
jx = ExaPF.Jacobian(polar, pflow ∘ basis, mapx)


#=
    Build preconditioner
=#
npartitions = 700
noverlap = 1
# Get reduced space Jacobian on the CPU
J = ExaPF.jacobian_sparsity(polar, pflow)
J = J[:, mapx]
precond = LS.BlockJacobiPreconditioner(J, npartitions, localdevice, noverlap)

#=
    Instantiate itertive linear solver
=#
lin_solver = ExaPF.LinearSolvers.KrylovBICGSTAB
algo = LS.KrylovBICGSTAB(J_gpu; P=precond)

#=
    Power flow resolution
=#
ExaPF.init!(polar, stack)
convergence = ExaPF.nlsolve!(pf_solver, jx, stack; linear_solver=algo)
