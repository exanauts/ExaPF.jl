ExaPF release notes
==================

Version 0.5.0 (XXX)
-----------------------------------

## PowerSystem

## AutoDiff

- New features
  * New AD `AutoDiff.Hessian` to compute the adjoint-Hessian-vector product of any function. (#99)
  * The submodule `AutoDiff` has been cleaned and its usage generalized. The constructor of `AutoDiff.Jacobian` has been moved in `src/Polar/derivatives.jl`. We could now compute the Jacobian of any constraint function implemented in `src/Polar/Constraints/`. (#106)
- Code changes
  * Fix the performance of the seeding by moving it onto the host (#102)

## PolarForm

- API changes
  * Add a new `NewtonRaphsonSolver` object, storing all parameters associated with the Newton-Raphson algorithm. (#88)
    The signature of the powerflow solver `ExaPF.powerflow` becomes:
    ```julia
    algo = ExaPF.NewtonRaphsonSolver(tol=1e-10, verbose=1)
    ExaPF.powerflow(polar, jacobian_autodiff, buffer, algo)
    ```
    instead of
    ```julia
    ExaPF.powerflow(polar, jacobian_autodiff, buffer; verbose=1, ε_tol=1e-10)
    ```
  * The constraints in `PolarForm` have been renamed, to improve clarity:
    ```julia
    residual_function -> power_balance
    state_constraint -> voltage_magnitude_constraints
    power_constraints -> {active_power_constraints,reactive_power_constraints}
    ```
- New features
  * Add support for lineflow constraints in the polar formulation. (#91)
    The constraint is implemented in `src/Polar/Constraints/line_flow.jl`, and shares the same signature as the other constraints:
    ```julia
    m = size_constraint(polar, flow_constraints)
    cons = zeros(m)
    flow_constraints(polar, cons, buffer)
    ```
    The function `flow_constraints` adds explicit bounds on the absolute value of the complex power flowing through the lines:
    ```
    |S_f|^2 <= S_max^2
    |S_t|^2 <= S_max^2
    ```
    leading to `2 * n_lines` additional constraints in the model.
  * Implement handwritten adjoints for polar formulation's functions (#107)
  * Add a new function `adjoint!` to compute the adjoint of any constraint. (#109)
  * We could now update inplace the load in `PolarForm`, with
    ```julia
    setvalues!(model, PS.ActiveLoad(), values)
    setvalues!(model, PS.ReactiveLoad(), values)
    ```
- Code changes
  * The definition of the polar formulation has been moved inside `src/Polar/`, instead of `src/model/Polar/`. Now, `src/model.jl` implements only the generic templates to define a new mathematical model.
  * The constraints corresponding to the polar formulation are now implemented in `src/Polar/Constraints`
    ```
    src/Polar/Constraints/
    ├── constraints.jl
    ├── active_power.jl
    ├── line_flow.jl
    ├── power_balance.jl
    ├── reactive_power.jl
    └── voltage_magnitude.jl
    ```

## Evaluators
- API changes
  * `ReducedSpaceEvaluator` no longer stores explicitly the state `x` in its attribute, but implicitly. Now, the attribute `buffer` stores all the values needed for the computation of the callback. (#82)
  * The constructor for `ReducedSpaceEvaluator` has been updated, from
    ```julia
    function ReducedSpaceEvaluator(
        model, x, u, p;
        constraints=Function[state_constraint, power_constraints],
        linear_solver=DirectSolver(),
        ε_tol=1e-12,
        verbose_level=VERBOSE_LEVEL_NONE,
    )
    ```
    to
    ```julia
    function ReducedSpaceEvaluator(
        model, x, u;
        constraints=Function[voltage_magnitude_constraints, active_power_constraints, reactive_power_constraints],
        linear_solver=DirectSolver(),
        powerflow_solver=NewtonRaphson(tol=1e-12),
    )
    ```
- New features
  * Add a new function `ExaPF.optimize!` to find the optimal solution of any problem specified as a `AbstractNLPEvaluator`, with signature:
    ```julia
    ExaPF.optimize!(optimizer, nlp)
    ```
    By default, `optimize!` supports `MathOptInterface` solvers out-of-the box:
    ```julia
    ExaPF.optimize!(Ipopt.Optimizer(), nlp)
    ```
    But other optimization procedures could be implemented as well.
  * Implement reduced Hessian for `ReducedSpaceEvaluator`, `SlackEvaluator` and `AugLagEvaluator`. (#93)
    The reduced Hessian is computed using AD (forward over reverse), with handcoded adjoints. The signature is:
    ```julia
    hessian!(nlp, H, u)  # evaluate Hessian inplace
    hessprod!(nlp, Hv, u, v)  # evaluate Hessian-vector product inplace
    ```
  * Add Hessian and Hessian-vector product in MOI wrapper (#100)
  * Bridge ExaPF with (ProxAL)[https://github.com/exanauts/ProxAL.jl]. A new evaluator `ProxALEvaluator` is introduced, allowing to formulate problems with (fixed) ramping constraints. (#76)
  * Add a new `SlackEvaluator` to reformulate automatically problems with inequality constraints with equality constraints, by adding additional non-negative slack variables. (#94)
  * Add a new `FeasibilityEvaluator` to automatically compute a feasible point for any `AbstractNLPEvaluator`.
- Fixes
  * Fix type inference on Julia 1.6-rc1. (#98)

## Maintenance

* Update the structure of the test directory to separate the different functionalities:
  ```
  test/
      ├── Evaluators
      │   ├── auglag.jl
      │   ├── interface.jl
      │   ├── MOI_wrapper.jl
      │   ├── powerflow.jl
      │   ├── proxal_evaluator.jl
      │   ├── reduced_evaluator.jl
      │   └── test_rgm.jl
      ├── iterative_solvers.jl
      ├── perf.jl
      ├── Polar
      │   ├── autodiff.jl
      │   ├── gradient.jl
      │   ├── hessian.jl
      │   ├── matpower.jl
      │   └── polar_form.jl
      ├── powersystem.jl
      └── runtests.jl
  ```
* Update the documentation (#101)
* Update `Krylov.jl` to version `0.6.0`. (#78)
* Update benchmark script
* Deprecated scripts have been removed. (#87)
* Fix typing of attributes in `BlockJacobiPreconditioner`


Version 0.4.0 (December 4, 2020)
-----------------------------------

* CUDA.jl 2.0 support
* New API
* Objective with handcoded adjoints
* Benchmark scripts (`benchmark/benchmark.sh`)
* Krylov.jl BiCGSTAB (`LinearSolvers.KrylovBICGSTAB`)
* Eigen BiCGSTAB (`LinearSolvers.EigenBICGSTAB`)
* New non-linear evaluators for penalty algorithms (`PenaltyEvaluator`) and Augmented Lagrangian algorithms (`AugLagEvaluator`)
* Reduced space gradient method (`scripts/dommel.jl`)
* Penalty and Augmented Lagrangian algorithms (`scripts/auglag.jl`)
* Updated documentation
