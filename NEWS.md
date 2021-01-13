ExaPF release notes
==================

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
