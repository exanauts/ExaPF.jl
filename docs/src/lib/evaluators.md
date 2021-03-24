```@meta
CurrentModule = ExaPF
```

# Evaluators

## Description

```@docs
AbstractNLPEvaluator
```


## API Reference

### Optimization
```@docs
optimize!
```

### Attributes
```@docs
Variables
Constraints
n_variables
n_constraints
constraints_type

```

### Utilities

```@docs
reset!
primal_infeasibility
primal_infeasibility!
```

## Callbacks

### Objective

```@docs
gradient!
hessprod!
hessian!

```

### Constraints

```@docs
constraint!
jacobian_structure!
jacobian!
jprod!
jtprod!
ojtprod!
```

### Second-order

```@docs
hessian_lagrangian_penalty_prod!

```

## ReducedSpaceEvaluator
When working in the reduced space, we could use
the corresponding `ReducedSpaceEvaluator`:
```@docs
ReducedSpaceEvaluator
```

## SlackEvaluator
```@docs
SlackEvaluator
```

## AugLagEvaluator

```@docs
AugLagEvaluator
```

## MOIEvaluator
The bridge to MathOptInterface is encoded by
the `MOIEvaluator` structure:
```@docs
MOIEvaluator
```

## ProxALEvaluator

```@docs
ProxALEvaluator
```

