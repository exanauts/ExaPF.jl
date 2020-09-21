```@meta
CurrentModule = ExaPF
DocTestSetup = quote
    using ExaPF
end
DocTestFilters = [r"ExaPF"]
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

## API Reference

```@docs
n_variables
n_constraints
objective
gradient!
constraint!
jacobian_structure!
jacobian!
hessian!

```
