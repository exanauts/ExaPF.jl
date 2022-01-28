export NewtonRaphson

abstract type AbstractNonLinearSolver end

"""
    NewtonRaphson <: AbstractNonLinearSolver

Newton-Raphson algorithm.

### Attributes
- `maxiter::Int` (default 20): maximum number of iterations
- `tol::Float64` (default `1e-8`): tolerance of the algorithm
- `verbose::Int` (default `0`): verbosity level

"""
struct NewtonRaphson <: AbstractNonLinearSolver
    maxiter::Int
    tol::Float64
    verbose::Int
end
NewtonRaphson(; maxiter=20, tol=1e-8, verbose=0) = NewtonRaphson(maxiter, tol, verbose)

"""
    ConvergenceStatus

Convergence status returned by a nonlinear algorithm.

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

@doc raw"""
    nlsolve!(
        algo::NewtonRaphson,
        jac::Jacobian,
        stack::NetworkStack;
        linear_solver=DirectSolver(jac.J),
        nl_buffer=NLBuffer(size(jac, 2)),
    )

Solve the nonlinear system of equations ``g(x) = 0`` with
a [`NewtonRaphson`](@ref) algorithm. At each iteration, we
update the variable ``x`` as
```math
    x_{k+1} = x_{k} - (∇g_k)^{-1} g(x_k)

```
till ``\| g(x_k) \| < ε_{tol}``

In the implementation,
- the function ``g`` is specified in `jac.func`,
- the initial variable ``x_0`` in `stack::NetworkStack` (with mapping `jac.map`),
- the Jacobian ``∇g`` is computed automatically in `jac`, with automatic differentiation.

Note that `stack` is modified inplace during the iterations of algorithm.

The Jacobian `jac` should be instantied before calling this function.
By default, the linear system ``(∇g_k)^{-1} g(x_k)`` is solved
using a LU factorization. You can specify a different linear solver
by changing the optional argument `linear_solver`.

## Arguments

* `algo::NewtonRaphon`: Newton-Raphson object, storing the options of the algorithm
* `jac::Jacobian`: Stores the function ``g`` and its Jacobian ``∇g``. The Jacobian is updated with automatic differentiation.
* `stack::NetworkStack`: initial values
* `linear_solver::AbstractLinearSolver`: linear solver used to compute the Newton step
* `nl_buffer::NLBuffer`: buffer storing the residual vector and the descent direction `Δx`. Can be reused to avoid unecessary allocations.

"""
function nlsolve!(
    algo::NewtonRaphson,
    jac::Jacobian,
    stack::NetworkStack{VT,Buf};
    linear_solver=DirectSolver(jac.J),
    nl_buffer=NLBuffer{VT}(size(jac, 2)),
) where {VT, Buf}
    iter = 0
    converged = false
    normF = Inf
    linsol_iters = Int[]

    map = jac.map
    x = view(stack.input, map)
    n = (jac.ncolors+1) * length(nl_buffer.y)
    F = reshape(reinterpret(eltype(nl_buffer.y), jac.t1sF), n)
    stridedF = @view F[1:jac.ncolors+1:n]
    residual = nl_buffer.y
    Δx = nl_buffer.x

    for i in 1:algo.maxiter
        J = jacobian!(jac, stack)
        copyto!(residual, stridedF)

        normF = xnorm(residual)
        if algo.verbose >= 1
            @printf("#it %d: %.5e\n", i-1, normF)
        end
        if xnorm(residual) < algo.tol
            converged = true
            break
        end

        # Update
        LS.update!(linear_solver, J)
        n_iters = LS.ldiv!(linear_solver, Δx, J, residual)
        x .= x .- Δx

        push!(linsol_iters, n_iters)

        iter += 1
    end
    return ConvergenceStatus(converged, iter, normF, sum(linsol_iters))
end

"""
    run_pf(
        polar::PolarForm, stack::NetworkStack;
        rtol=1e-8, max_iter=20, verbose=0,
    )

Solve the power flow equations ``g(x, u) = 0`` w.r.t. the stack ``x``,
using the ([`NewtonRaphson`](@ref) algorithm.
The initial state ``x`` is specified implicitly inside
`stack`, with the mapping [`my_map`](@ref) associated to the polar
formulation. The object `stack` is modified inplace in the function.

The algorithm stops when a tolerance `rtol` or a maximum number of
iterations `maxiter` is reached.

## Arguments

* `polar::AbstractFormulation`: formulation of the power flow equation
* `stack::NetworkStack`: initial values in the network

"""
function run_pf(
    polar::PolarForm, stack::NetworkStack;
    rtol=1e-8, max_iter=20, verbose=0,
)
    solver = NewtonRaphson(tol=rtol, maxiter=max_iter, verbose=verbose)
    mapx = my_map(polar, State())

    basis = PolarBasis(polar)
    func = PowerFlowBalance(polar) ∘ basis
    jac = Jacobian(polar, func, mapx)

    conv = nlsolve!(solver, jac, stack)
    return conv
end

