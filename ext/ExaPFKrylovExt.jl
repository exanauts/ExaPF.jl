module ExaPFKrylovExt

using ExaPF
using SparseArrays
import Krylov
using KrylovPreconditioners

const LS = ExaPF.LinearSolvers
const KP = KrylovPreconditioners

LS.update!(is::LS.AbstractIterativeLinearSolver, J::AbstractMatrix) = KP.update!(is.precond, J)

function LS.Dqgmres(J::AbstractSparseMatrix;
    P=BlockJacobiPreconditioner(J), memory=4, verbose=false
)
    n, m = size(J)
    S = LS._get_type(J)
    workspace = Krylov.DqgmresWorkspace(n, m, S; memory)
    return LS.Dqgmres(workspace, P, memory, verbose)
end

function LS.ldiv!(solver::LS.Dqgmres,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector; options...
)
    Krylov.dqgmres!(solver.inner, J, x; N=solver.precond)
    copyto!(y, solver.inner.x)
    return Krylov.iteration_count(solver.inner)
end

function LS.Bicgstab(J::AbstractSparseMatrix;
    P=BlockJacobiPreconditioner(J), verbose=0, rtol=1e-10, atol=1e-10, ldiv=false, scaling=false, maxiter=size(J,1)
)
    n, m = size(J)
    S = LS._get_type(J)
    workspace = Krylov.BicgstabWorkspace(n, m, S)
    return LS.Bicgstab(workspace, P, verbose, atol, rtol, ldiv, scaling, maxiter)
end

function LS.ldiv!(solver::LS.Bicgstab,
    y::AbstractVector, J::AbstractMatrix, x::AbstractVector;
    max_atol = solver.atol, max_rtol = solver.rtol, options...
)

    atol = max_atol < solver.atol ? solver.atol : max_atol
    rtol = max_rtol < solver.rtol ? solver.rtol : max_rtol
    Krylov.bicgstab!(
        solver.inner, J, x;
        N=solver.precond,
        atol=atol,
        rtol=rtol,
        verbose=solver.verbose,
        history=true,
        ldiv=solver.ldiv,
        itmax=solver.maxiter,
    )
    if solver.inner.stats.status == "breakdown αₖ == 0"
        @warn("BICGSTAB failed to converge. Final status is $(solver.inner.stats.status)")
    end
    copyto!(y, solver.inner.x)
    return Krylov.iteration_count(solver.inner)
end

# KrylovPreconditioners implements scaling_csr! for GPU CSR matrices only.
function LS.scaling!(::LS.Bicgstab, A::AbstractSparseMatrix, b)
    KP.scaling_csr!(A, b)
end

end
