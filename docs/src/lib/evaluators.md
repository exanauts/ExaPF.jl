```@meta
CurrentModule = ExaPF
```

# Evaluators

## Description

```@docs
AbstractNLPEvaluator
```

When working in the reduced space, we could use
the corresponding `ReducedSpaceEvaluator`:
```@docs
ReducedSpaceEvaluator
```

The bridge to MathOptInterface is encoded by
the `MOIEvaluator` structure:
```@docs
MOIEvaluator
```

## API Reference

### Attributes
```@docs
Variables
Constraints
n_variables
n_constraints

```

### Callbacks
```@docs
objective
gradient!
constraint!
jacobian_structure!
jacobian!
jtprod!
hessian!

```

### Utilities

```@docs
reset!
primal_infeasibility
primal_infeasibility!
```
