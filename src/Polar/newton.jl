export NewtonRaphson

abstract type AbstractNonLinearSolver end

"""
    NewtonRaphson <: AbstractNonLinearSolver

Newton-Raphson algorithm. Used to solve the non-linear equation
``g(x, u) = 0``, at a fixed control ``u``.

### Attributes
- `maxiter::Int` (default 20): maximum number of iterations
- `tol::Float64` (default `1e-8`): tolerance of the algorithm
- `verbose::Int` (default `NONE`): verbosity level

"""
struct NewtonRaphson <: AbstractNonLinearSolver
    maxiter::Int
    tol::Float64
    verbose::Int
end
NewtonRaphson(; maxiter=20, tol=1e-8, verbose=0) = NewtonRaphson(maxiter, tol, verbose)

"""
    ConvergenceStatus

Convergence status returned by a non-linear algorithm.

### Attributes
- `has_converged::Bool`: states whether the algorithm has converged.
- `n_iterations::Int`: total number of iterations of the non-linear algorithm.
- `norm_residuals::Float64`: final residual.
- `n_linear_solves::Int`: number of linear systems ``Ax = b`` resolved during the run.

"""
struct ConvergenceStatus
    has_converged::Bool
    n_iterations::Int
    norm_residuals::Float64
    n_linear_solves::Int
end

struct NLBuffer{VT}
    x::VT
    y::VT
end
NLBuffer{VT}(n::Int) where VT = NLBuffer(VT(undef, n), VT(undef, n))


function extract_values!(dest, src)
    @assert length(dest) == length(src)
    for i in eachindex(dest)
        dest[i] = src[i].value
    end
end

function nlsolve!(
    algo::NewtonRaphson,
    jac::MyJacobian,
    state::NetworkStack{VT,Buf};
    linear_solver=DirectSolver(),
    nl_buffer=NLBuffer{VT}(size(jac, 2)),
) where {VT, Buf}
    iter = 0
    converged = false
    normF = Inf
    linsol_iters = Int[]

    map = jac.map
    x = view(state.input, map)

    residual = nl_buffer.y
    Δx = nl_buffer.x

    for i in 1:algo.maxiter
        J = jacobian!(jac, state)
        extract_values!(residual, jac.t1sF)

        normF = xnorm(residual)
        if xnorm(residual) < algo.tol
            converged = true
            break
        end

        # Update
        n_iters = LS.ldiv!(linear_solver, Δx, J, residual)
        x .= x .- Δx

        push!(linsol_iters, n_iters)

        iter += 1
    end
    return ConvergenceStatus(converged, iter, normF, sum(linsol_iters))
end

"""
    run_pf(polar::PolarForm, stack::NetworkStack;
           rtol=1e-8, max_iter=20,
    )

Solve the power flow equations ``g(x, u) = 0`` w.r.t. the state ``x``,
using the ([`NewtonRaphson`](@ref) algorithm.
The initial state ``x`` is specified inside
`stack`. The object `stack` is modified inplace in the function.

The algorithm stops when a tolerance `rtol` or a maximum number of
iterations `maxiter` is reached.

## Arguments

* `polar::AbstractFormulation`: formulation of the power flow equation
* `stack::NetworkStack`: initial values in the network

"""
function run_pf(
    polar::PolarForm, state::NetworkStack;
    rtol=1e-8, max_iter=20,
)
    solver = NewtonRaphson(tol=rtol, maxiter=max_iter)
    mapx = my_map(polar, State())

    basis = PolarBasis(polar)
    func = PowerFlowBalance(polar) ∘ basis
    jac = MyJacobian(polar, func, mapx)

    conv = nlsolve!(solver, jac, state)
    return conv
end

