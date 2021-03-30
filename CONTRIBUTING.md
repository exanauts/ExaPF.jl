# Contributing to ExaPF

Welcome to ExaPF! All help is welcome to improve the current state of ExaPF.
If you are interested in contributing to ExaPF, this document details the
development procedure we are using.

## Bug reports and feature requests

The [issue tracker](https://github.com/exanauts/ExaPF.jl/issues)
tracks all the bugs and feature requests. If you find a bug in ExaPF,
and/or you would like to add a new feature, you are welcome to open a
new issue ticket!

## Contributing to the code base

### Workflow
Our development workflow is based on a mildly simplified version of [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow). Thus, we have at least two active branches

- `master` which is the lastest stable release of ExaPF,
- and `develop` which is the current development branch, and as is, could be unstable.

At each release, the `develop` branch is merged into `master` through a `release` branch
before tagging the next release. While in the `release` branch we check that

- the unit tests passed,
- we haven't introduced any performance regression (this is tested in the folder `ExaPF/benchmark/`),
- the documentation is building properly, and is up-to-date.

### Structure of the code

The code is split in four parts:

- `ExaPF/PowerSystem/`: in this directory, we implement everything related to network's data (MATPOWER's parsers, specification of the incidence matrix, etc.)
- `ExaPF/LinearSolvers/`: this directory implements wrappers to various linear solvers, both direct and iterative, on the CPU and on the GPU. All linear solvers should overload the `LinearSolvers.ldiv!` function as an interface to solve linear system.
- `ExaPF/PolarForm/` implements the polar formulation of the network equations.
- `ExaPF/Evaluators/` implements various evaluators objects, allowing interfacing ExaPF with different optimization solvers to solve the Optimal Power Flow (OPF) problem.

### Style guide

ExaPF is following closely the [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/),
as extended by the [JuMP style guide](https://jump.dev/JuMP.jl/stable/style/).
In each pull-request, we will ensure that the following requirements
are fulfilled:

- all new features are tested with proper unit-tests.
- all functions exposed in the API are documented properly.
- the code style is consistent (we are using 4-space indent!).

