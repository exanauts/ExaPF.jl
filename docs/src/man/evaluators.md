# Evaluators

In `ExaPF.jl`, the evaluators are the final layer of the structure.
They take as input a given `AbstractFormulation` and implement the
callbacks for the optimization solvers.

## Overview

An `AbstractEvaluator` is tailored to the optimization problem
associated with an underlying `AbstractFormulation`:
```math
\begin{aligned}
\min_{x, u} \;          & f(x, u)     \\
\text{subject to} \quad & g(x, u) = 0 \\
                        & h(x, u) \leq 0.
\end{aligned}
```
In this problem, we recognize the state $x$ and the control $u$ introduced earlier.
The objective function $f(x, u)$, the equality constraints $g(x, u) = 0$
and the inequality constraints $h(x, u) \leq 0$ all depend on the
state $x$ and the control $u$. The non-linear
functions are all specified inside the `AbstractFormulation`.


## Callbacks

Most non-linear optimization algorithms rely on *callbacks* to pass
information about the structure of the problem to the optimizer.
The evaluator allows to have a proper splitting
between the model (formulated in the `AbstractFormulation` layer)
and the optimization algorithms. By design, the implementation
of an `AbstractEvaluator` is similar in spirit than the implementations
introduced in other packages, as

- MathOptInterface.jl's [AbstractNLPEvaluator](https://jump.dev/MathOptInterface.jl/stable/apireference/#MathOptInterface.AbstractNLPEvaluator)
- NLPModels' [AbstractNLPModel](https://juliasmoothoptimizers.github.io/NLPModels.jl/stable/api/#AbstractNLPModel-functions)

The evaluator caches internally all the information needed to evaluate
the callbacks. Once a new point $u$ passed to the evaluator,
a function `update!` is being called to update all the internal structures,
according to the model specified in the underlying `AbstractFormulation`.
Also, this cache allows to reduce the number of memory allocations to
its minimum. In a sense, the evaluator is equivalent to Julia's closures,
but tailored to `ExaPF.jl` usage.


## A journey in the reduced space

When we aim at optimizing the problem directly in the reduced
space manifold, the `ReducedSpaceNLPEvaluator` is our workhorse.
We recall that the reduced space is defined implicitly by the
powerflow equations:
```math
    g(x(u), u) = 0.
```
By design, the `ReducedSpaceNLPEvaluator` works in the powerflow
manifold $(x(u), u)$. Hence, the reduced optimization problem
writes out
```math
\begin{aligned}
\min_{u} \;          & f(x(u), u)     \\
\text{subject to} \quad & h(x(u), u) \leq 0.
\end{aligned}
```
This formulation comes with two advantages:

- if the dimension of the state is large, the reduced problem has
  a lower dimension rendering it more amenable for the optimization algorithm.
- the powerflow equality constraints $g(x, u) = 0$ disappear in the reduced problem.

### Playing with the ReducedSpaceNLPEvaluator

#### Constructor
To create a `ReducedSpaceNLPEvaluator`, we need a given formulation
`form::AbstractFormulation`, together with an initial control `u0`,
an initial state `x0` and a vector of parameters `p`:
```julia-repl
julia> nlp = ExaPF.ReducedSpaceEvaluator(polar, x0, u0, p)

```
By default, `nlp` has no inequality constraint. The user could
specified them in the constructor:
```julia-repl
julia> constraints = Function[ExaPF.state_constraints, ExaPF.power_constraints]
julia> nlp = ExaPF.ReducedSpaceEvaluator(polar, x0, u0, p, constraints=constraints)

```

!!! note
    If `form` is defined on the GPU, `ReducedSpaceNLPEvaluator`
    will work automatically on the GPU too.


#### Caching

To juggle between the mathematical description (characterized
by a state $x$ and a control $u$) and the physical description (characterized
by the voltage and the power injection at each bus), the evaluator `nlp`
stores internally a cache `nlp.network_cache`, with type `AbstractPhysicalCache`.

#### Evaluation of the callbacks

Now that we have a `nlp` evaluator available, we could embed it in any
optimization routine. For instance, suppose we have a new control `uk`
available. First, we need to find the corresponding state `xk`
in the powerflow manifold. In the evaluator's API, this sums up to:
```julia-repl
ExaPF.update!(nlp, uk)
```
The function `update!` will
- Feed the physical description `nlp.network_cache` with the values stored in the new control `uk`.
- Solve the powerflow equations corresponding to the formulation specified in `form`. This operation
  updates the cache `nlp.network_cache` inplace.
- Update internally the state `x`.

Once the function `update!` called (and only once), we could evaluate
all the different callbacks independently from one other.
```julia-repl
julia> cost = ExaPF.objective(nlp, uk)
# Evaluate objective's gradient
julia> g = similar(uk)
julia> fill!(g, 0)
julia> ExaPF.gradient!(nlp, g, uk)
# Evaluate constraints
julia> cons = zeros(n_constraints(nlp))
julia> ExaPF.constraint!(nlp, cons, uk)
## Evaluate Jacobian
julia> ExaPF.jacobian!(nlp, jac, uk)
```


## Going back to the full space

In the long term, we are planning to implement a `FullSpaceNLPEvaluator` as well.

