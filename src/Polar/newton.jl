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
    time_jacobian::Float64
    time_linear_solver::Float64
    time_total::Float64
end

function Base.show(io::IO, conv::ConvergenceStatus)
    println(io, "Power flow has converged: ", conv.has_converged)
    @printf(io, "  * #iterations: %d\n", conv.n_iterations)
    @printf(io, "  * Time Jacobian (s) ........: %1.4f\n", conv.time_jacobian)
    @printf(io, "  * Time linear solver (s) ...: %1.4f\n", conv.time_linear_solver)
    @printf(io, "  * Time total (s) ...........: %1.4f\n", conv.time_total)
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

### Arguments

* `algo::NewtonRaphon`: Newton-Raphson object, storing the options of the algorithm
* `jac::Jacobian`: Stores the function ``g`` and its Jacobian ``∇g``. The Jacobian is updated with automatic differentiation.
* `stack::NetworkStack`: initial values
* `linear_solver::AbstractLinearSolver`: linear solver used to compute the Newton step
* `nl_buffer::NLBuffer`: buffer storing the residual vector and the descent direction `Δx`. Can be reused to avoid unecessary allocations.

### Examples
```jldoctest; setup=:(using ExaPF)
julia> polar = ExaPF.load_polar("case9");

julia> powerflow = ExaPF.PowerFlowBalance(polar) ∘ ExaPF.PolarBasis(polar);

julia> jx = ExaPF.Jacobian(polar, powerflow, State());

julia> stack = ExaPF.NetworkStack(polar);

julia> conv = ExaPF.nlsolve!(NewtonRaphson(verbose=1), jx, stack);
#it 0: 2.64764e+00
#it 1: 2.03366e-01
#it 2: 2.94166e-03
#it 3: 8.85300e-07
#it 4: 7.53857e-14

julia> conv.has_converged
true

```
"""
function nlsolve!(
    algo::NewtonRaphson,
    jac::AutoDiff.AbstractJacobian,
    stack::AbstractNetworkStack{VT};
    linear_solver=DirectSolver(jac.J),
    nl_buffer=NLBuffer{VT}(size(jac.J, 2)),
) where {VT}
    iter = 0
    converged = false
    normF = Inf
    linsol_iters = Int[]
    time_total = 0.0
    time_jacobian = 0.0
    time_linear_solver = 0.0

    map = jac.map
    x = view(stack.input, map)
    n = (jac.ncolors+1) * length(nl_buffer.y)
    F = reshape(reinterpret(eltype(nl_buffer.y), jac.t1sF), n)
    stridedF = @view F[1:jac.ncolors+1:n]
    residual = nl_buffer.y
    Δx = nl_buffer.x

    tic = time()
    for i in 1:algo.maxiter
        time_jacobian += @elapsed begin
            J = jacobian!(jac, stack)
        end
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
        time_linear_solver += @elapsed begin
            LS.update!(linear_solver, J)
            n_iters = LS.ldiv!(linear_solver, Δx, J, residual)
        end
        x .= x .- Δx
        push!(linsol_iters, n_iters)

        iter += 1
    end
    time_total = time() - tic
    return ConvergenceStatus(
        converged, iter, normF, sum(linsol_iters), time_jacobian, time_linear_solver, time_total,
    )
end

"""
    run_pf(
        polar::PolarForm, stack::NetworkStack;
        rtol=1e-8, max_iter=20, verbose=0,
    )

Solve the power flow equations ``g(x, u) = 0`` w.r.t. the stack ``x``,
using the ([`NewtonRaphson`](@ref) algorithm.
The initial state ``x`` is specified implicitly inside
`stack`, with the mapping [`mapping`](@ref) associated to the polar
formulation. The object `stack` is modified inplace in the function.

The algorithm stops when a tolerance `rtol` or a maximum number of
iterations `maxiter` is reached.

### Arguments

* `polar::AbstractFormulation`: formulation of the power flow equation
* `stack::NetworkStack`: initial values in the network

### Examples
```jldoctest; setup=:(using ExaPF)
julia> polar = ExaPF.load_polar("case9");

julia> stack = ExaPF.NetworkStack(polar);

julia> conv = run_pf(polar, stack; verbose=1);
#it 0: 2.64764e+00
#it 1: 2.03366e-01
#it 2: 2.94166e-03
#it 3: 8.85300e-07
#it 4: 7.53857e-14

julia> conv.has_converged
true

```
"""
function run_pf(
    polar::PolarForm, stack::NetworkStack;
    rtol=1e-8, max_iter=20, verbose=0,
)
    solver = NewtonRaphson(tol=rtol, maxiter=max_iter, verbose=verbose)
    mapx = mapping(polar, State())

    basis = PolarBasis(polar)
    func = PowerFlowBalance(polar) ∘ basis
    jac = Jacobian(polar, func, mapx)

    conv = nlsolve!(solver, jac, stack)
    return conv
end

