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
or we could alternatively instantiate the evaluator directly from
a MATPOWER (or PSSE) instance:
```julia-repl
julia> datafile = "case9.m"
julia> nlp = ExaPF.ReducedSpaceEvaluator(datafile)
A ReducedSpaceEvaluator object
    * device: KernelAbstractions.CPU()
    * #vars: 5
    * #cons: 10
    * constraints:
        - state_constraint
        - power_constraints
    * linear solver: ExaPF.LinearSolvers.DirectSolver()

```

Let's describe the output of the last command.

* `device: KernelAbstractions.CPU()`: the evaluator is instantiated on the CPU ;
* `#vars: 5`: it has 5 optimization variables ;
* `#cons: 10`: the problem has 10 inequality constraints ;
* `constraints:` by default, `nlp` has two inequality constraints: `state_constraint` (specifying the bounds $x_L \leq x(u) \leq x_U$ on the state $x$) and `power_constraints` (bounding the active and reactive power of the generators) ;
* `linear solver: ExaPF.LinearSolvers.DirectSolver()`: to solve the linear systems, the evaluator uses a direct linear algebra solver.

Of course, these settings are only specified by default, and the user is free
to choose the parameters she wants. For instance,

* we could remove all constraints by passing an empty array of constraints
  to the evaluator:
  ```julia-repl
  julia> constraints = Function[]
  julia> nlp = ExaPF.ReducedSpaceEvaluator(datafile; constraints=constraints)
  ```
* we could load the evaluator on the GPU simply by changing the device:
  ```julia-repl
  julia> nlp = ExaPF.ReducedSpaceEvaluator(datafile; device=CUDADevice())
  ```



#### Caching

To juggle between the mathematical description (characterized
by a state $x$ and a control $u$) and the physical description (characterized
by the voltage and power injection at each bus), the evaluator `nlp`
stores internally a cache `nlp.buffer`, with type `AbstractNetworkBuffer`.

#### Evaluation of the callbacks

Now that we have a `nlp` evaluator available, we could embed it in any
optimization routine. For instance, suppose we have a new control `uk`
available. First, we need to find the corresponding state `xk`
in the powerflow manifold. In the evaluator's API, this sums up to:
```julia-repl
ExaPF.update!(nlp, uk)
```
The function `update!` will
- Feed the physical description `nlp.buffer` with the values stored in the new control `uk`.
- Solve the powerflow equations corresponding to the formulation specified in `form`. This operation
  updates the cache `nlp.buffer` inplace.
- Update internally the state `x`.

Once the function `update!` called (and only once), we could evaluate
all the different callbacks independently from one other.

* Objective
  ```julia-repl
  julia> cost = ExaPF.objective(nlp, uk)
  ```
* Objective's gradient
  ```julia-repl
  julia> g = zeros(n_variables(nlp))
  julia> ExaPF.gradient!(nlp, g, uk)
  ```
* Constraints
  ```julia-repl
  # Evaluate constraints
  julia> cons = zeros(n_constraints(nlp))
  julia> ExaPF.constraint!(nlp, cons, uk)
  ```
* Constraints' Jacobian
  ```julia-repl
  ## Evaluate Jacobian
  julia> ExaPF.jacobian!(nlp, jac, uk)
  ```
* Constraints' transpose Jacobian-vector product
  ```julia-repl
  ## Evaluate transpose Jacobian-vector product
  julia> v = zeros(n_constraints(nlp))
  julia> jv = zeros(n_variables(nlp))
  julia> ExaPF.jtprod!(nlp, jv, uk, v)
  ```

!!! note
    Once the powerflow equations solved in a `update!` call, the solution `x` is stored in memory in the attribute `nlp.x`. The state `x` will be used as a starting point for the next resolution of powerflow equations.


## Passing the problem to an optimization solver with MathOptInterface

`ExaPF.jl` provides a utility to pass the non-linear structure
specified by a `AbstractNLPEvaluator` to a `MathOptInterface` (MOI)
optimization problem. That allows to solve the corresponding
optimal power flow problem using any non-linear optimization solver compatible
with MOI.

For instance, we could solve the reduced problem specified
in `nlp` with Ipopt in a few lines of code:

```julia
using Ipopt
optimizer = Ipopt.Optimizer()

block_data = MOI.NLPBlockData(nlp)

u♭, u♯ = ExaPF.bounds(nlp, ExaPF.Variables())
u0 = ExaPF.initial(nlp)
n = ExaPF.n_variables(nlp)
vars = MOI.add_variables(optimizer, n)

# Set bounds and initial values
for i in 1:n
    MOI.add_constraint(
        optimizer,
        MOI.SingleVariable(vars[i]),
        MOI.LessThan(u♯[i])
    )
    MOI.add_constraint(
        optimizer,
        MOI.SingleVariable(vars[i]),
        MOI.GreaterThan(u♭[i])
    )
    MOI.set(optimizer, MOI.VariablePrimalStart(), vars[i], u0[i])
end

MOI.set(optimizer, MOI.NLPBlock(), block_data)
MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
MOI.optimize!(optimizer)

solution = (
    minimum=MOI.get(optimizer, MOI.ObjectiveValue()),
    minimizer=[MOI.get(optimizer, MOI.VariablePrimal(), v) for v in vars],
)
MOI.empty!(optimizer)
```


## Going back to the full space

In the long term, we are planning to implement a `FullSpaceNLPEvaluator` as well.

