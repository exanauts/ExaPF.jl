module LinearSolvers

using LinearAlgebra
using Adapt

using Printf
using SparseArrays

import Base: show

using KernelAbstractions

import ..ExaPF: xnorm
import ..ExaPF.AutoDiff: AbstractJacobian

import Base.size, Base.sizeof, Base.format_bytes

using KLU

const KA = KernelAbstractions

export list_solvers, default_linear_solver
export DirectSolver, Bicgstab, Dqgmres
export do_scaling, scaling!

@enum(
    SolveStatus,
    Unsolved,
    MaxIterations,
    NotANumber,
    Converged,
    Diverged,
)

abstract type AbstractLinearSolver end
abstract type AbstractIterativeLinearSolver <: AbstractLinearSolver end

"""
    list_solvers(::KernelAbstractions.Device)

List linear solvers available on current backend (CPU, NVIDIA GPU, AMD GPU).

"""
function list_solvers end

"""
    ldiv!(solver, x, A, y)
    ldiv!(solver, x, y)

* `solver::AbstractLinearSolver`: linear solver to solve the system
* `x::AbstractVector`: Solution
* `A::AbstractMatrix`: Input matrix
* `y::AbstractVector`: RHS

Solve the linear system ``A x = y`` using the algorithm
specified in `solver`. If `A` is not specified, the function
will used directly the factorization stored inside `solver`.

"""
function ldiv! end

"""
    rdiv!(solver, x, A, y)
    rdiv!(solver, x, y)

* `solver::AbstractLinearSolver`: linear solver to solve the system
* `x::AbstractVector`: Solution
* `A::AbstractMatrix`: Input matrix
* `y::AbstractVector`: RHS

Solve the linear system ``A^⊤ x = y`` using the algorithm
specified in `solver`. If `A` is not specified, the function
will used directly the factorization stored inside `solver`.

"""
function rdiv! end

_get_type(J) = error("No handling of sparse Jacobian type defined in LinearSolvers")
_get_type(J::SparseMatrixCSC) = Vector{Float64}
do_scaling(linear_solver) = false
scaling!(A, b) = nothing

"""
    DirectSolver <: AbstractLinearSolver
    DirectSolver(A; kwargs...)

Solve linear system ``A x = b`` with a direct sparse linear solver.

* On `CPU`, `DirectSolver` redirects the resolution to KLU if `A` is a `SparseMatrixCSC`.
* On CUDA GPU, `DirectSolver` redirects the resolution to cuDSS if `A` is a `CuSparseMatrixCSR`.
"""
struct DirectSolver{Fac<:LinearAlgebra.Factorization} <: AbstractLinearSolver
    factorization::Fac
end

function DirectSolver(J::SparseMatrixCSC; kwargs...)
    klu_solver = klu(J)
    ds = DirectSolver(klu_solver)
    return ds
end

function update!(ds::DirectSolver, J::AbstractMatrix)
    klu!(ds.factorization, J) # Update factorization inplace
end

# Reuse factorization in update
function ldiv!(ds::DirectSolver{<:LinearAlgebra.Factorization}, y::AbstractVector, J::AbstractMatrix, x::AbstractVector; options...)
    LinearAlgebra.ldiv!(y, ds.factorization, x) # Forward-backward solve
    return 0
end

# Solve system Ax = y
function ldiv!(ds::DirectSolver{<:LinearAlgebra.Factorization}, y::AbstractArray, x::AbstractArray; options...)
    LinearAlgebra.ldiv!(y, ds.factorization, x) # Forward-backward solve
    return 0
end

function ldiv!(ds::DirectSolver{<:LinearAlgebra.Factorization}, y::AbstractArray; options...)
    LinearAlgebra.ldiv!(ds.factorization, y) # Forward-backward solve
    return 0
end

# Solve system A'x = y
function rdiv!(ds::DirectSolver{<:LinearAlgebra.Factorization}, y::AbstractArray, x::AbstractArray)
    LinearAlgebra.ldiv!(y, ds.factorization', x) # Forward-backward solve
    return 0
end

# The iterative solvers wrap Krylov.jl and KrylovPreconditioners.jl, which are
# weak dependencies: their constructors and methods live in ext/ExaPFKrylovExt.jl.
_has_krylov() = !isnothing(Base.get_extension(parentmodule(@__MODULE__), :ExaPFKrylovExt))

"""
    Dqgmres <: AbstractIterativeLinearSolver
    Dqgmres(J; P=BlockJacobiPreconditioner(J), memory=4, verbose=false)

Wrap `Krylov.jl`'s DQGMRES algorithm to solve iteratively the linear system
``A x = y``.

Requires `Krylov.jl` and `KrylovPreconditioners.jl` to be loaded.
"""
struct Dqgmres{W, Pr} <: AbstractIterativeLinearSolver
    inner::W
    precond::Pr
    memory::Int
    verbose::Bool
end

"""
    Bicgstab <: AbstractIterativeLinearSolver
    Bicgstab(J; P=BlockJacobiPreconditioner(J), verbose=0, rtol=1e-10, atol=1e-10, ldiv=false, scaling=false, maxiter=size(J, 1))

Wrap `Krylov.jl`'s BICGSTAB algorithm to solve iteratively the linear system
``A x = y``.

Requires `Krylov.jl` and `KrylovPreconditioners.jl` to be loaded.
"""
struct Bicgstab{W, Pr} <: AbstractIterativeLinearSolver
    inner::W
    precond::Pr
    verbose::Int
    atol::Float64
    rtol::Float64
    ldiv::Bool
    scaling::Bool
    maxiter::Int64
end
do_scaling(linear_solver::Bicgstab) = linear_solver.scaling

"""
    list_solvers(::KA.CPU)

List all linear solvers available for solving the (batch) power flow on the CPU.
"""
list_solvers(::KA.CPU) = _has_krylov() ? [DirectSolver, Dqgmres, Bicgstab] : [DirectSolver]

"""
    default_linear_solver(A::SparseMatrixCSC; nblocks::Int=1)

Return the default linear solver for CPU with a sparse matrix.
Uses DirectSolver (KLU) as the default.
"""
function default_linear_solver(A::SparseMatrixCSC; nblocks::Int=1)
    return DirectSolver(A)
end

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        f = exc.f === Core.kwcall ? exc.args[2] : exc.f
        if (f === Bicgstab || f === Dqgmres) && !_has_krylov()
            print(io, "\n$(f) requires Krylov.jl and KrylovPreconditioners.jl. ",
                  "Load them with `using Krylov, KrylovPreconditioners`.")
        end
    end
end

end
