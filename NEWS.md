ExaPF release notes
===================

Version 0.12 (December 12th, 2025)
----------------------------------
- API changes
  * New `PowerFlowProblem` struct exposed to users
  * Extend `run_pf` function to run power flow or batch power flow directly from a MATPOWER case file (#310)
  * Rename `ArrowheadJacobian` to `BatchJacobian` (#298)
  * Rename `ArrowheadHessian` to `BatchHessian` (#302)
- New features
  * Add batch powerflow API (#308)
  * Add support for uniform batches with CUDSS.jl v0.6.2 (#306)
  * Initial batch support implementation (#303)
- Improvements
  * Use CUDSS.jl by default in the CUDA extension (#294)
  * Remove CUSOLVERRF and add CUDSS to documentation (#291)
  * Support Krylov.jl v0.10 (#290)
  * Julia 1.12 support and maintenance (#295)
- Bug fixes
  * Fix AMDGPU.jl extension

Version 0.11 (December 5th, 2023)
---------------------------------
- Remove the block-GMRES implementation
- Remove custom BICGSTAB implementations
- Use KrylovPreconditioners.jl for the block-Jacobi preconditioner
- Add scaling for Krylov methods
- Use KLU per default
- Added new benchmark scripts for linear solvers

Version 0.10 (November 1st, 2023)
---------------------------------
- Adding extension support for CUDA.jl and AMDGPU.jl
- Adding block-GMRES implementation

Version 0.9.3 (September 27th, 2023)
---------------------------------
- Add support for SCOPF problem (#274)

Version 0.9.2 (September 25th, 2023)
---------------------------------
- Add support to CUDA.jl 5.*

Version 0.9.0 (April 19th, 2023)
---------------------------------
- Update to KernelAbstractions 0.9 with respective API changes
- Update to Krylov.jl 0.9, and CUDA.jl 4.1

Version 0.8.0 (August 16th, 2022)
---------------------------------

- API changes
  * Add new `BlockPolarForm` structure encoding block-structured power grid model
- Bug fixes & improvements
  * add function `load_case` to import data directly from ExaData (#257)
  * test `ArrowheadJacobian` correctness with FiniteDiff (#263)
  * remove all custom SpMM kernels (#264)
- Documentation
  * Add new tutorials in ExaPF for batch power flow and linear solver (#258)
  * Add doctests in expressionsn (#256 #261)

Version 0.7.2 (May 31th, 2022)
------------------------------

- Bug fixes & improvements
  * Fix unitinialized arrays in `NetworkStack` (#248)
  * Fix finite-difference checks in tests (#251)

Version 0.7.1 (May 9th, 2022)
-----------------------------

- New features
  * Add `BlockNetworkStack` structure for batch evaluation (#241)
  * Add `ArrowheadJacobian` and `ArrowheadHessian` for stochastic OPF (#241)
  * define loads as parameters inside `NetworkStack` (#238)
- Bug fixes & improvements
  * Migrate to KernelAbstractions 0.8 (#236)
  * Drop support for Julia 1.6
  * Migrate to Krylov.jl 0.8 (#239)
  * Update CI

Version 0.7.0 (March 1st, 2022)
-------------------------------

- API changes
    * The interface has been completely revamped to [vectorize all operations](https://frapac.github.io/pages/notes/exapf/) (#217)
- New features
    * Add overlapping Schwarz-preconditioner (#220)
    * Full support for multi-generators (#217 #223)
    * Add new benchmark suite (#226)

Version 0.6.0 (November 19th, 2021)
-----------------------------------

- New features
  * ExaPF now supports multiple generators per bus [experimental]
  * ExaPF now implements a batch power flow solver [experimental]
  * Add new functions `network_operation` and `power_injection`
  * Tests directory has been updated, and all tests are now implemented inside dedicated modules (#156).
  This allows to test directly any MATPOWER instance, simply as
  ```julia
  TestPolarFormulation(datafile, CPU(), Array)
  ```
  * Add benchmark script for power flow solver
- API changes
  * All `AbstractNLPEvaluator` structures have been moved to [ExaOpt](https://github.com/exanauts/ExaPF-Opt) (#191)
- Bug Fixes & Improvements
  * ExaPF now supports `CUDA.jl 3.*`
  * `IterativeSolvers.jl` has been removed from the deps
  * Names have been homogenized in `PolarNetworkState`.
  * Fix power flow solver when phase shift is activated (#183)
  * ExaPF now supports `CUDA.allowscalar(false)` (#181)

Version 0.5.0 (April 14, 2021)
------------------------------

## AutoDiff

- New features
  * New AD `AutoDiff.Hessian` to compute the adjoint-Hessian-vector product of any function. (#99)
  * The submodule `AutoDiff` has been cleaned and its usage generalized. The constructor of `AutoDiff.Jacobian` has been moved in `src/Polar/derivatives.jl`. We could now compute the Jacobian of any constraint function implemented in `src/Polar/Constraints/`. (#106)
- Code changes
  * Fix the performance of the seeding by moving it onto the host (#102)

## PolarForm

- API changes
  * Add a new `NewtonRaphson` object, storing all parameters associated with the Newton-Raphson algorithm. (#88)
    The signature of the powerflow solver `ExaPF.powerflow` becomes:
    ```julia
    algo = ExaPF.NewtonRaphson(tol=1e-10, verbose=1)
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
        model;
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
* Update the documentation (#101 #134 #149)
* Update the README.md to include a quickstart guide
* Update CI to use Julia 1.6 by default
* Update `Krylov.jl` to version `0.6.0`. (#78)
* Update `KernelAbstractions.jl` to version `0.5`.
* Update benchmark script
* Deprecated scripts have been removed. (#87)
* Fix typing of attributes in `BlockJacobiPreconditioner`
* Fix non-deterministic bug issued by a wrong initilization of KA kernels (#145)


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
