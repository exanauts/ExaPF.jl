# Evaluators

In `ExaPF.jl`, the evaluators are the final layer of the structure.
They take as input a given `AbstractFormulation` and implement the
callbacks for the optimization solvers.

## Overview of the AbstractNLPEvaluator

An `AbstractNLPEvaluator` implements an optimization problem
associated with an underlying `AbstractFormulation`:
```math
\begin{aligned}
\min_{u \in \mathbb{R}^n} \;             & f(u)     \\
\text{subject to} \quad & g(u) = 0 \\
                        & h(u) \leq 0.
\end{aligned}
```
with $f: \mathbb{R}^n \to \mathbb{R}$ the objective function,
$g: \mathbb{R}^n \to \mathbb{R}^{m_E}$ non-linear equality constraints and
$h: \mathbb{R}^n \to \mathbb{R}^{m_I}$ non-linear inequality constraints.

### Callbacks

Most non-linear optimization algorithms rely on *callbacks* to pass
information about the structure of the problem to the optimizer.
In `ExaPF`, the implementation of the evaluators allows to have a proper splitting
between the model (formulated in the `AbstractFormulation` layer)
and the optimization algorithms. By design, the implementation
of an `AbstractEvaluator` shares a similar spirit with the implementations
introduced in other packages, as

- MathOptInterface.jl's [AbstractNLPEvaluator](https://jump.dev/MathOptInterface.jl/stable/apireference/#MathOptInterface.AbstractNLPEvaluator)
- NLPModels' [AbstractNLPModel](https://juliasmoothoptimizers.github.io/NLPModels.jl/stable/api/#AbstractNLPModel-functions)

Internally, the evaluator caches all the information needed to evaluate
the callbacks (e.g. the polar representation of the problem, with voltage
magnitudes and angles). This cache allows to reduce the number of memory allocations to
its minimum.
Once a new variable $u$ passed to the evaluator
a function `update!` is being called to update the cache,
according to the model specified in the underlying `AbstractFormulation`.
Denoting by `nlp` an instance of AbstractNLPEvaluator, the cache is
updated via
```julia-repl
julia> ExaPF.update!(nlp, u)
```

Once the internal structure updated, we are ready to call the different
callbacks, in every order. For instance, computing the objective, the
gradient and the constraints amounts to
```julia-repl
# Objective
julia> obj = ExaPF.objective(nlp, u)
# Gradient
julia> g = zeros(n_variables(nlp))
julia> ExaPF.gradient!(nlp, g, u)
# Constraints
julia> cons = zeros(n_constraints(nlp))
julia> ExaPF.constraint!(nlp, cons, u)

```


## A journey to the reduced space with the ReducedSpaceEvaluator

When we aim at optimizing the problem directly in the powerflow
manifold, the `ReducedSpaceEvaluator` is our workhorse.
We recall that the powerflow manifold is defined implicitly by the
powerflow equations:
```math
    g(x(u), u) = 0.
```
By design, the `ReducedSpaceEvaluator` works in the reduced
space $(x(u), u)$. Hence, the reduced optimization problem writes out
```math
\begin{aligned}
\min_{u \in \mathbb{R}^n} \; & f(x(u), u) \\
\text{subject to} \quad      & h(x(u), u) \leq 0.
\end{aligned}
```
This formulation comes with two advantages:

- if the dimension of the state is large, the reduced problem has
  a lower dimension.
- the powerflow equality constraints $g(x, u) = 0$ disappear in the reduced problem.


### Playing with the ReducedSpaceEvaluator

#### Constructor
To create a `ReducedSpaceEvaluator`, we need a given polar formulation
`polar::PolarForm`, together with an initial control `u0`:
```julia-repl
julia> nlp = ExaPF.ReducedSpaceEvaluator(polar, u0)

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
        - voltage_magnitude_constraints
        - active_power_constraints
        - reactive_power_constraints
    * linear solver: ExaPF.LinearSolvers.DirectSolver()

```

Let's describe the output of the last command.

* `device: KernelAbstractions.CPU()`: the evaluator is instantiated on the CPU ;
* `#vars: 5`: it has 5 optimization variables ;
* `#cons: 10`: and 10 inequality constraints ;
* `constraints:` by default, `nlp` comes with three inequality constraints: `voltage_magnitude_constraints` (specifying the bounds $x_L \leq x(u) \leq x_U$ on the state $x$), `active_power_constraints` and `reactive_power_constraints` (bounding the active and reactive power of the generators) ;
* `linear solver: ExaPF.LinearSolvers.DirectSolver()`: to solve the linear systems, the evaluator uses a direct linear algebra solver.

Of course, these settings are only specified by default. The user is free
to choose the parameters she wants. For instance,

* We could remove all constraints by passing an empty array of constraints
  to the evaluator:
  ```julia-repl
  julia> constraints = Function[]
  julia> nlp = ExaPF.ReducedSpaceEvaluator(datafile; constraints=constraints)
  ```
* We could load the evaluator on the GPU simply by changing the `device` option:
  ```julia-repl
  julia> nlp = ExaPF.ReducedSpaceEvaluator(datafile; device=CUDADevice())
  ```



#### Caching

To juggle between the mathematical description (characterized
by a state $x$ and a control $u$) and the physical description (characterized
by the voltage and power injection at each bus), the evaluator `nlp`
stores internally a cache `nlp.buffer`, with type `AbstractNetworkBuffer`.
```julia-repl
julia> buffer = get(nlp, ExaPF.PhysicalState())
```

#### Evaluation of the callbacks

Now that we have a `nlp` evaluator available, we can embed it in any
optimization routine. For instance, suppose we have a new control `uk`
available. First, we need to find the corresponding state `xk`,
such that ``g(x_k, u_k) = 0``.
In the evaluator's API, this sums up to:
```julia-repl
ExaPF.update!(nlp, uk)

```
The function `update!` will
- Feed the physical description `nlp.buffer` with the values stored in the new control `uk`.
- Solve the powerflow equations corresponding to the formulation specified in `form`. This operation
  updates the cache `nlp.buffer` inplace.

Once the function `update!` called (and only after that), we can evaluate
all the different callbacks, independently of one other.

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
* Constraints' Jacobian-vector product:
  ```julia-repl
  ## Evaluate Jacobian-vector product
  julia> v = zeros(n_variables(nlp))
  julia> jv = zeros(n_constraints(nlp))
  julia> ExaPF.jprod!(nlp, jv, uk, v)
  ```
* Constraints' transpose Jacobian-vector product
  ```julia-repl
  ## Evaluate transpose Jacobian-vector product
  julia> v = zeros(n_constraints(nlp))
  julia> jv = zeros(n_variables(nlp))
  julia> ExaPF.jtprod!(nlp, jv, uk, v)
  ```
* Hessian-vector product:
  ```julia-repl
  ## Evaluate transpose Jacobian-vector product
  julia> v = zeros(n_variables(nlp))
  julia> hv = zeros(n_variables(nlp))
  julia> ExaPF.hessprod!(nlp, hv, uk, v)
  ```
* Hessian:
  ```julia-repl
  ## Evaluate transpose Jacobian-vector product
  julia> H = zeros(n_variables(nlp), n_variables(nlp))
  julia> ExaPF.hessprod!(nlp, H, uk)
  ```

!!! note
    Once the powerflow equations solved in a `update!` call, the solution ``x_k`` is stored implicitly in `nlp.buffer`. These values will be used as a starting point for the next resolution of powerflow equations.

## Passing the problem to an optimization solver with MathOptInterface

`ExaPF.jl` provides a utility to pass the non-linear structure
specified by a `AbstractNLPEvaluator` to a `MathOptInterface` (MOI)
optimization problem. That allows to solve the corresponding
optimal power flow problem using any non-linear optimization solver compatible
with MOI.

For instance, we can solve the reduced problem specified
in `nlp` with Ipopt. In a few lines of code:

```julia
using Ipopt
optimizer = Ipopt.Optimizer()
MOI.set(optimizer, MOI.RawParameter("print_level"), 5)
MOI.set(optimizer, MOI.RawParameter("limited_memory_max_history"), 50)
MOI.set(optimizer, MOI.RawParameter("hessian_approximation"), "limited-memory")
solution = ExaPF.optimize!(optimizer, nlp)
MOI.empty!(optimizer)

```
