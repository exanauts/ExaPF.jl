using LazyArtifacts
using SparseArrays

using KernelAbstractions
using CUDA

using ExaPF
const LS = ExaPF.LinearSolvers

const INSTANCES_DIR = joinpath(artifact"ExaData", "ExaData")

USEGPU = 0


if USEGPU == 0
    localdevice = CPU()
else
    localdevice = CUDABackend()
end

case = "case1354pegase.m"
casefile = joinpath(INSTANCES_DIR, case)

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
pf_solver = NewtonRaphson(tol=1e-6, verbose=0)
# Expressions
basis = ExaPF.PolarBasis(polar)
pflow = ExaPF.PowerFlowBalance(polar)
# Init AD
jx = ExaPF.Jacobian(polar, pflow ∘ basis, mapx)


#=
    Build preconditioner
=#
npartitions = 64
# NB: Use noverlap > 0 only on the GPU
noverlap = 0
# Get reduced space Jacobian on the CPU
V = ExaPF.voltage(stack)
J = ExaPF.matpower_jacobian(polar, pflow, V)
J = J[:, mapx]
precond = LS.BlockJacobiPreconditioner(J, npartitions, localdevice, noverlap)

#=
    Instantiate iterative linear solver
=#
lin_solver = ExaPF.LinearSolvers.KrylovBICGSTAB
algo = LS.KrylovBICGSTAB(J; P=precond)

#=
    Power flow resolution
=#
ExaPF.init!(polar, stack)
convergence = ExaPF.nlsolve!(pf_solver, jx, stack; linear_solver=algo)

@assert convergence.has_converged
